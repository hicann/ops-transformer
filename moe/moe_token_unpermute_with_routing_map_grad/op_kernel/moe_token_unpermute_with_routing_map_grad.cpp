/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_unpermute_with_routing_map_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_true.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_false.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_true.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_false.h"

using namespace MoeTokenUnpermuteWithRoutingMapGrad;

extern "C" __global__ __aicore__ void moe_token_unpermute_with_routing_map_grad(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR out_index, GM_ADDR permute_token_id, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(10)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadTrue<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
#ifdef DTYPE_PROBS
    if (TILING_KEY_IS(1)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadFalse<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(11)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadTrue<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
#endif
}