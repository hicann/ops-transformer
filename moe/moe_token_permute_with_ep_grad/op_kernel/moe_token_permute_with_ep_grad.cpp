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
 * \file moe_token_permute_with_ep_grad.cpp
 * \brief
 */

#include "moe_token_permute_with_ep_grad.h"
#include "kernel_operator.h"

#if !defined(DTYPE_PERMUTED_TOKENS_OUTPUT_GRAD)
#define DTYPE_PERMUTED_TOKENS_OUTPUT_GRAD bfloat16_t
#endif
#if !defined(DTYPE_PERMUTED_PROBS_OUTPUT_GRAD)
#define DTYPE_PERMUTED_PROBS_OUTPUT_GRAD DTYPE_PERMUTED_TOKENS_OUTPUT_GRAD
#endif

#define GENERAL_OP_IMPL(opClass, tokenDtype, probsDtype, haveProbs) \
    do { \
        opClass<tokenDtype, int32_t, tokenDtype, false, true> op1; \
        op1.Init(permuted_output_grad, sorted_indices, nullptr, token_grad, tiling_data, &t_pipe); \
        op1.Process(); \
        if (haveProbs) { \
            t_pipe.Destroy(); \
            TPipe t_probs_pipe; \
            opClass<probsDtype, int32_t, probsDtype, false, false> op2; \
            op2.Init(permuted_probs_output_grad, sorted_indices, nullptr, probs_grad, tiling_data, &t_probs_pipe); \
            op2.Process(); \
        } \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_permute_with_ep_grad(GM_ADDR permuted_output_grad,
                                                                     GM_ADDR sorted_indices,
                                                                     GM_ADDR permuted_probs_output_grad,
                                                                     GM_ADDR token_grad, GM_ADDR probs_grad,
                                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data_in, tiling);
    const MoeTokenPermuteWithEpGradTilingData* __restrict tiling_data = &tiling_data_in;
    TPipe t_pipe;
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(KernelMoeTokenUnpermuteWithEp, DTYPE_PERMUTED_TOKENS_OUTPUT_GRAD,
                        DTYPE_PERMUTED_PROBS_OUTPUT_GRAD, false);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(KernelMoeTokenUnpermuteWithEp, DTYPE_PERMUTED_TOKENS_OUTPUT_GRAD,
                        DTYPE_PERMUTED_PROBS_OUTPUT_GRAD, true);
    }
}
