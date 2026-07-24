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
 * \file inplace_fused_causal_conv1d_apt.cpp
 * \brief InplaceFusedCausalConv1d kernel entry point (reuses FusedCausalConv1d kernel implementations)
 */

#if __has_include("../../fused_causal_conv1d/op_kernel/arch35/fused_causal_conv1d_cut_bsh.h")
#include "../../fused_causal_conv1d/op_kernel/arch35/fused_causal_conv1d_cut_bsh.h"
#include "../../fused_causal_conv1d/op_kernel/arch35/fused_causal_conv1d_cut_bh.h"
#else
#include "../fused_causal_conv1d/arch35/fused_causal_conv1d_cut_bsh.h"
#include "../fused_causal_conv1d/arch35/fused_causal_conv1d_cut_bh.h"
#endif

#define TILING_KEY_BSH_BF16 10000
#define TILING_KEY_BSH_FP16 10001
#define TILING_KEY_BH_BF16 20000
#define TILING_KEY_BH_FP16 20001

using namespace FusedCausalConv1dCutBSHNs;
extern "C" __global__ __aicore__ void
inplace_fused_causal_conv1d(GM_ADDR x,                           // 输入0: x
                            GM_ADDR weight,                      // 输入1: weight
                            GM_ADDR convStates,                  // 输入2: convStates
                            GM_ADDR queryStartLoc,               // 输入3: queryStartLoc (optional)
                            GM_ADDR cacheIndices,                // 输入4: cacheIndices (optional)
                            GM_ADDR initialStateMode,            // 输入5: initialStateMode (optional)
                            GM_ADDR bias,                        // 输入6: bias (optional)
                            GM_ADDR numAcceptedToken,            // 输入7: numAcceptedToken (optional)
                            GM_ADDR numComputedTokens,           // 输入8: numComputedTokens (optional, APC)
                            GM_ADDR blockIdxFirstScheduledToken, // 输入9: blockIdxFirstScheduledToken (optional, APC)
                            GM_ADDR blockIdxLastScheduledToken,  // 输入10: blockIdxLastScheduledToken (optional, APC)
                            GM_ADDR initialStateIdx,             // 输入11: initialStateIdx (optional, APC)
                            GM_ADDR outputConvStates,            // 输出0: convStates output
                            GM_ADDR outputX,                     // 输出1: x (inplace write-back)
                            GM_ADDR workspace,                   // workspace
                            GM_ADDR tiling)                      // tiling
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FusedCausalConv1dCutBSHTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR >= 20000)", FusedCausalConv1dCutBHTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_BH_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(FusedCausalConv1dCutBHTilingData, tilingData, tiling);
        FusedCausalConv1dCutBH<bfloat16_t> op(&pipe);
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices, numAcceptedToken, numComputedTokens,
                blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, outputX, &tilingData);
        op.Process();
    }

    if (TILING_KEY_IS(TILING_KEY_BH_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(FusedCausalConv1dCutBHTilingData, tilingData, tiling);
        FusedCausalConv1dCutBH<half> op(&pipe);
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices, numAcceptedToken, numComputedTokens,
                blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, outputX, &tilingData);
        op.Process();
    }

    if (TILING_KEY_IS(TILING_KEY_BSH_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(FusedCausalConv1dCutBSHTilingData, tilingData, tiling);
        FusedCausalConv1dCutBSH<bfloat16_t> op;
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices, numAcceptedToken, numComputedTokens,
                blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, outputX, workspace,
                &tilingData);
        op.Process();
    }

    if (TILING_KEY_IS(TILING_KEY_BSH_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(FusedCausalConv1dCutBSHTilingData, tilingData, tiling);
        FusedCausalConv1dCutBSH<half> op;
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices, numAcceptedToken, numComputedTokens,
                blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, outputX, workspace,
                &tilingData);
        op.Process();
    }
}
