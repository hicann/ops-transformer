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
 * \file moe_token_unpermute_with_ep.cpp
 * \brief
 */
#include "moe_token_unpermute_with_ep.h"
#include "kernel_operator.h"

#if !defined(DTYPE_PERMUTED_TOKENS)
#define DTYPE_PERMUTED_TOKENS bfloat16_t
#endif
#if !defined(DTYPE_PROBS)
#define DTYPE_PROBS DTYPE_PERMUTED_TOKENS
#endif

#define MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(T1, T2, T3, haveProbs, isUnpermute) \
    do { \
        KernelMoeTokenUnpermuteWithEp<T1, T2, T3, haveProbs, isUnpermute> op; \
        op.Init(permuted_tokens, sorted_indices, probs, unpermuted_tokens, tiling_data, &t_pipe); \
        op.Process(); \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_unpermute_with_ep(GM_ADDR permuted_tokens, GM_ADDR sorted_indices,
                                                                  GM_ADDR probs, GM_ADDR unpermuted_tokens,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data_in, tiling);
    const MoeTokenUnpermuteWithEpTilingData* __restrict tiling_data = &tiling_data_in;
    TPipe t_pipe;
    if (TILING_KEY_IS(0)) {
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, DTYPE_PROBS, false, true);
    } else if (TILING_KEY_IS(1)) {
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, float, true, true);
    } else if (TILING_KEY_IS(2)) {
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, half, true, true);
    } else if (TILING_KEY_IS(3)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, bfloat16_t, true, true);
#endif
    }
}
