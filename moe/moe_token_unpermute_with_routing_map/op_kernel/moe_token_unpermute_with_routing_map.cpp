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
 * \file moe_token_unpermute_with_routing_map.cpp
 * \brief
 */

#include "moe_token_unpermute_with_routing_map_pad.h"
#include "moe_token_unpermute_with_routing_map_not_pad.h"
#include "masked_select.h"
#include "kernel_operator.h"

#if !defined(DTYPE_PERMUTED_TOKENS)
#define DTYPE_PERMUTED_TOKENS bfloat16_t
#endif
#if !defined(DTYPE_PROBS)
#define DTYPE_PROBS DTYPE_PERMUTED_TOKENS
#endif

#define GENERAL_OP_IMPL() \
    do { \
        auto t = &tilingData; \
        TPipe sortPipe; \
        KernelMaskedSelectV3<DTYPE_PERMUTE_PROBS> opMS; \
        auto maskedSelectTilingData = &(t->maskedSelectParamsOp); \
        opMS.Init(probs, routing_map, permute_probs, userWS, maskedSelectTilingData, &sortPipe); \
        opMS.Process(permute_probs); \
        sortPipe.Destroy(); \
        AscendC::SyncAll(); \
    } while (0)

#define MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_IMPL(T1, T2, T3, haveProbs) \
    do { \
        auto t = &tilingData; \
        TPipe sortPipe; \
        GET_TILING_DATA(tiling_data_in, tiling); \
        const MoeTokenUnpermuteWithRoutingMapTilingData* __restrict tiling_data = &tiling_data_in; \
        MoeTokenUnpermuteWithRoutingMap::KernelMoeTokenUnpermuteWithRoutingMap<T1, T2, T3, haveProbs> op; \
        op.Init(permuted_tokens, sorted_indices, permute_probs, unpermuted_tokens, tiling_data); \
        op.Process(); \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_unpermute_with_routing_map(
    GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR routing_map, GM_ADDR probs, GM_ADDR unpermuted_tokens,
    GM_ADDR out_index, GM_ADDR permute_token_id, GM_ADDR permute_probs, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    //==================================BF16==================================
    if (TILING_KEY_IS(1000)) {
        MoeTokenUnpermuteWithRoutingMap::KernelMoeTokenUnpermuteWithRoutingMapPad<DTYPE_PERMUTE_PROBS> op;
        op.Init(sorted_indices, probs, permute_probs, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL();
        MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, DTYPE_PERMUTE_PROBS, true);
    } else if (TILING_KEY_IS(0)) {
        MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, DTYPE_PERMUTE_PROBS, false);
    }
}
