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
 * \file moe_token_unpermute_with_ep_grad.cpp
 * \brief
 */
#include "moe_token_unpermute_grad_prob_none_with_ep.h"
#include "moe_token_unpermute_grad_prob_not_none_with_ep.h"
using namespace AscendC;
using namespace MoeTokenUnpermuteWithEpGrad;
extern "C" __global__ __aicore__ void moe_token_unpermute_with_ep_grad(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices, GM_ADDR permuted_tokens, GM_ADDR probs,
    GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    // 0: padded_mode = False, 不存在prob
    // 1: padded_mode = False, 存在prob
    // 10: padded_mode = True, 不存在prob
    // 11: padded_mode = True, 存在prob
    if (TILING_KEY_IS(0) || TILING_KEY_IS(10)) {
        MoeTokenUnpermuteGradProbNone<DTYPE_PERMUTED_TOKENS, int32_t> moeTokenUnpermuteGradProbNoneOp;
        moeTokenUnpermuteGradProbNoneOp.Init(
            permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad,
            tilingData);
        moeTokenUnpermuteGradProbNoneOp.Process();
    }
#ifdef DTYPE_PROBS
    if (TILING_KEY_IS(1) || TILING_KEY_IS(11)) {
        MoeTokenUnpermuteGradProbNotNone<DTYPE_PERMUTED_TOKENS, int32_t, DTYPE_PROBS>
            moeTokenUnpermuteGradProbNotNoneOp;
        moeTokenUnpermuteGradProbNotNoneOp.Init(
            permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad,
            tilingData);
        moeTokenUnpermuteGradProbNotNoneOp.Process();
    }
#endif
}