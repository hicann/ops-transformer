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
 * \file moe_re_routing.cpp
 * \brief
 */

#include "moe_re_routing.h"
#include "kernel_operator.h"

#define MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT32_SCALES_FLOAT32 100000UL
#define MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT64_SCALES_FLOAT32 100010UL
#define MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT32_SCALES_FLOAT32 100100UL
#define MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT64_SCALES_FLOAT32 100110UL
#define MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT32_SCALES_FLOAT32 100200UL
#define MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT64_SCALES_FLOAT32 100210UL

extern "C" __global__ __aicore__ void moe_re_routing(GM_ADDR tokens, GM_ADDR expertTokensNumPerRank,
                                                     GM_ADDR perTokenScales, GM_ADDR permuteTokens,
                                                     GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx,
                                                     GM_ADDR expertTokenNum, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT32_SCALES_FLOAT32)) {
        KernelMoeReRouting<int8_t, int32_t, float> op(&pipe, &tiling_data);
        op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens, permutePerTokenScales, permuteTokenIdx,
                expertTokenNum);
        op.Process();
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT64_SCALES_FLOAT32)) {
        KernelMoeReRouting<int8_t, int64_t, float> op(&pipe, &tiling_data);
        op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens, permutePerTokenScales, permuteTokenIdx,
                expertTokenNum);
        op.Process();
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT32_SCALES_FLOAT32)) {
        KernelMoeReRouting<half, int32_t, float> op(&pipe, &tiling_data);
        op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens, permutePerTokenScales, permuteTokenIdx,
                expertTokenNum);
        op.Process();
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT64_SCALES_FLOAT32)) {
        KernelMoeReRouting<half, int64_t, float> op(&pipe, &tiling_data);
        op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens, permutePerTokenScales, permuteTokenIdx,
                expertTokenNum);
        op.Process();
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT32_SCALES_FLOAT32)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelMoeReRouting<bfloat16_t, int32_t, float> op(&pipe, &tiling_data);
        op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens, permutePerTokenScales, permuteTokenIdx,
                expertTokenNum);
        op.Process();
#endif
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT64_SCALES_FLOAT32)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelMoeReRouting<bfloat16_t, int64_t, float> op(&pipe, &tiling_data);
        op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens, permutePerTokenScales, permuteTokenIdx,
                expertTokenNum);
        op.Process();
#endif
    }
}
