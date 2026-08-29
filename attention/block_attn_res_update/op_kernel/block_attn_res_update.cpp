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
#include "arch35/block_attn_res_update_full_d.h"
#include "arch35/block_attn_res_update_tiling_key.h"

// The argument order follows the OpDef input/output order. ACLNNGraph aliases partial_block_ref with partial_block for
// reference launches. The kernel updates partial_block in place and does not access partial_block_ref or workspace.
template <bool SINGLE_TILE>
__global__ __aicore__ void block_attn_res_update(GM_ADDR partial_block, GM_ADDR delta, GM_ADDR pseudo_query,
                                                 GM_ADDR numerator, GM_ADDR logit_max, GM_ADDR exp_sum,
                                                 GM_ADDR partial_block_ref, GM_ADDR h, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    AscendC::InitSocState();
    (void)partial_block_ref;
    (void)workspace;
    REGISTER_NONE_TILING;
    GET_TILING_DATA_WITH_STRUCT(BlockAttnResUpdateTilingData, tilingData, tiling);
    {
        BlockAttnResUpdateOps::BlockAttnResUpdateFullD<SINGLE_TILE> op;
        op(partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, h, &tilingData);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
