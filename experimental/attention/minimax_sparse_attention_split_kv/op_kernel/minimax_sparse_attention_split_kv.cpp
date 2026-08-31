/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "minimax_sparse_attention_split_kv_tilingkey.h"
#include "minimax_sparse_attention_split_kv_kernel_interface.cpp"

extern "C" __global__ __aicore__ void minimax_sparse_attention_split_kv(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *k2qRowPtr, __gm__ uint8_t *k2qQIndices, __gm__ uint8_t *k2qSlotIndices,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    if (TILING_KEY_VAR >= MINIMAX_SA_SPLIT_KV_BASE_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
        TILING_KEY_IS(MINIMAX_SA_SPLIT_KV_BF16_D128_TILING);
#if TILING_KEY_VAR == MINIMAX_SA_SPLIT_KV_BF16_D128_TILING
        // innerPrecise==4 (default): bf16 softmax S + fp32 O_partial.
        MinimaxSaSplitKvInferIntf<bfloat16_t, bfloat16_t, float>(query, key, value, blockTable, k2qRowPtr, k2qQIndices,
                                                                 k2qSlotIndices, actualSeqLengths, actualSeqLengthsKv,
                                                                 attentionOut, softmaxLse, user, tiling);
#endif
        TILING_KEY_IS(MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_LOW_TILING);
#if TILING_KEY_VAR == MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_LOW_TILING
        // innerPrecise==1: REDtype(bf16) => PV fixpipe F322BF16 + Phase2 regbase cast
        // (bf16 O_partial). fp32 path (above) stays byte-identical.
        MinimaxSaSplitKvInferIntf<bfloat16_t, bfloat16_t, bfloat16_t>(
            query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths, actualSeqLengthsKv,
            attentionOut, softmaxLse, user, tiling);
#endif
        TILING_KEY_IS(MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_HIGH_TILING);
#if TILING_KEY_VAR == MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_HIGH_TILING
        // innerPrecise==0: SMDtype(float) => QK fixpipe NoQuant fp32 S + fp32 softmax;
        // REDtype(float) => fp32 O_partial (same Phase2 path as innerPrecise==4).
        MinimaxSaSplitKvInferIntf<bfloat16_t, float, float>(query, key, value, blockTable, k2qRowPtr, k2qQIndices,
                                                            k2qSlotIndices, actualSeqLengths, actualSeqLengthsKv,
                                                            attentionOut, softmaxLse, user, tiling);
#endif
        TILING_KEY_IS(MINIMAX_SA_SPLIT_KV_FP8_D128_BF16_TILING);
#if TILING_KEY_VAR == MINIMAX_SA_SPLIT_KV_FP8_D128_BF16_TILING
        // FP8 Q/K/V, bf16 softmax S, fp32 O_partial, bf16 attentionOut.
        MinimaxSaSplitKvInferIntf<fp8_e4m3fn_t, bfloat16_t, float>(
            query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths, actualSeqLengthsKv,
            attentionOut, softmaxLse, user, tiling);
#endif
#endif
    }
}
