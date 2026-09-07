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
 * \file block_attention_residuals_apt.cpp
 * \brief BlockAttentionResiduals
 */
#include "arch35/block_attention_residuals_reload.h"
#include "arch35/block_attention_residuals_resident.h"
#include "arch35/block_attention_residuals_hslice.h"
#include "block_attention_residuals_tiling_data.h"
#include "tiling_key_block_attention_residuals.h"

using namespace AscendC;
using namespace BlockAttentionResiduals;

// 输入 dtype 由编译期宏注入（框架按 def 的 DataType 组合为每个 dtype 生成独立二进制，
// 并以 -DDTYPE_PARTIAL_BLOCK=<type> 编译，参考 moe_token_permute）；未注入时默认 BF16。
#if !defined(DTYPE_PARTIAL_BLOCK)
#define DTYPE_PARTIAL_BLOCK bfloat16_t
#endif

extern "C" __global__ __aicore__ void block_attention_residuals(GM_ADDR partialBlock, GM_ADDR blockRes,
                                                                GM_ADDR projWeight, GM_ADDR normWeight,
                                                                GM_ADDR hiddenStates, GM_ADDR invNorm, GM_ADDR probs,
                                                                GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(BlockAttentionResidualsTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspaceGM; // 无用户 FP32-v Workspace；仅系统预留

    TPipe pipe;
    BlockAttentionResidualsInitParams initParams{partialBlock, blockRes, projWeight, normWeight,
                                                 hiddenStates, invNorm,  probs};

    // 按 tiling key 分派算法分支（RELOAD/RESIDENT/HSLICE）；dtype 由编译期宏 DTYPE_PARTIAL_BLOCK 决定
    if (TILING_KEY_IS(TILING_KEY_RELOAD)) {
        BlockAttentionResidualsReload<DTYPE_PARTIAL_BLOCK> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RESIDENT)) {
        BlockAttentionResidualsResident<DTYPE_PARTIAL_BLOCK> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HSLICE)) {
        BlockAttentionResidualsHSlice<DTYPE_PARTIAL_BLOCK> op(&pipe, &tilingData);
        op.Init(initParams);
        op.Process();
    }
}
