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
 * \file moe_token_permute_grad.cpp
 * \brief
 */

#include "moe_token_permute_grad.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void moe_token_permute_grad(GM_ADDR permuted_output_grad, GM_ADDR sorted_indices,
                                                             GM_ADDR input_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data_in, tiling);
    const MoeTokenPermuteGradTilingData* __restrict tiling_data = &tiling_data_in;

    // 调用unpermute计算， probs始终false
    if (TILING_KEY_IS(0)) {
        KernelMoeTokenUnpermute<bfloat16_t, int32_t, bfloat16_t, false> op;
        op.Init(permuted_output_grad, sorted_indices, nullptr, input_grad, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelMoeTokenUnpermute<half, int32_t, half, false> op;
        op.Init(permuted_output_grad, sorted_indices, nullptr, input_grad, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelMoeTokenUnpermute<float, int32_t, float, false> op;
        op.Init(permuted_output_grad, sorted_indices, nullptr, input_grad, tiling_data);
        op.Process();
    }
}
