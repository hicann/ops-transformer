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
 * \file block_attention_residuals_grad_apt.cpp
 * \brief block_attention_residuals_grad A5 (ascend950) kernel entry using arch35 regbase
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "block_attention_residuals_grad_tiling_data.h"
#include "block_attention_residuals_grad_tiling_key.h"
#include "arch35/block_attention_residuals_grad_regbase.h"
#include "arch35/block_attention_residuals_grad_split_h.h"

using namespace AscendC;
using namespace NsBlockAttentionResidualsGrad;

template <uint32_t hMode>
__global__ __aicore__ void block_attention_residuals_grad(GM_ADDR partial_block, GM_ADDR block_res, GM_ADDR proj_weight,
                                                          GM_ADDR norm_weight, GM_ADDR grad_hidden_states,
                                                          GM_ADDR inv_norm, GM_ADDR probs, GM_ADDR grad_partial_block,
                                                          GM_ADDR grad_block_res, GM_ADDR grad_proj_weight,
                                                          GM_ADDR grad_norm_weight, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BlockAttentionResidualsGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(BlockAttentionResidualsGradTilingData, tilingData, tiling);

    if constexpr (hMode == TPL_H_MODE_FULL) {
        TPipe pipe;
        BlockAttentionResidualsGradRegbase<DTYPE_PARTIAL_BLOCK> op(&pipe, &tilingData);
        op.Init(partial_block, block_res, proj_weight, norm_weight, grad_hidden_states, inv_norm, probs,
                grad_partial_block, grad_block_res, grad_proj_weight, grad_norm_weight, workspace);
        op.Process();
    } else if constexpr (hMode == TPL_H_MODE_SPLIT) {
        // A5 SPLIT_H 与 A2/A3 保持同一三阶段数学流程：
        // 完整 H 归约生成 meta -> 分 tile 写 gradV/权重部分和 -> 跨核归约。
        BlockAttentionResidualsGradA5SplitH<DTYPE_PARTIAL_BLOCK> op;
        op.Init(partial_block, block_res, proj_weight, norm_weight, grad_hidden_states, inv_norm, probs,
                grad_partial_block, grad_block_res, grad_proj_weight, grad_norm_weight, workspace, &tilingData);
        op.Process();
    }
}
