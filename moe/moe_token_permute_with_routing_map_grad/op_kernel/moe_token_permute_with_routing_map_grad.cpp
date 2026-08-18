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
 * \file moe_token_permute_with_routing_map_grad.cpp
 * \brief
 */

#include "moe_token_unpermute_routing_map_grad.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_token_permute_with_routing_map_grad_kernel.h"

extern "C" __global__ __aicore__ void moe_token_permute_with_routing_map_grad(
    GM_ADDR permutedTokenOutPutGrad, GM_ADDR permutedProbsOutPutGradOptional, GM_ADDR sortedIndices,
    GM_ADDR routingMapOptional, GM_ADDR tokensGradOut, GM_ADDR probsGradOutOptional, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(
        MoeTokenPermuteWithRoutingMapGradTilingData, moeTokenPermuteWithRoutingMapGradUnpermuteTilingData, moe_token_unpermute_tiling_data_in,
        tiling);
    GET_TILING_DATA_MEMBER(
        MoeTokenPermuteWithRoutingMapGradTilingData, moeTokenpermuteWithRoutingMapDropPadTilingData,
        drop_tiling_data_in, tiling);
    // 调用unpermute计算， probs始终false
    if (TILING_KEY_IS(0)) {
        KernelMoeTokenUnpermute<DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD, int32_t, DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD, false> op;
        op.Init(permutedTokenOutPutGrad, sortedIndices, nullptr, tokensGradOut, &moe_token_unpermute_tiling_data_in);
        op.Process();
    } else if (TILING_KEY_IS(1000)) {
        KernelMoeTokenPermuteWithRoutingMap<DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD, int32_t, DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD>
            op;
        op.Init(permutedProbsOutPutGradOptional, sortedIndices, probsGradOutOptional, &drop_tiling_data_in);
        op.Process();
    }
    return;
}
