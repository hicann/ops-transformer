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
 * \file block_attn_res_prepare_apt.cpp
 * \brief Kernel entry for BlockAttnResPrepare on Ascend 950.
 */

#include "arch35/block_attn_res_prepare_apt_tiling_key.h"
#include "arch35/block_attn_res_prepare_mix.h"
#include "arch35/block_attn_res_prepare_vector.h"

template <uint32_t TEMPLATE_MODE>
__global__ __aicore__ void block_attn_res_prepare(GM_ADDR blockRes, GM_ADDR validBlocks, GM_ADDR pseudoQuery,
                                                  GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_NONE_TILING;
    if constexpr (TEMPLATE_MODE == BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR) {
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareTilingData, tilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareVector op(&tilingData);
        op.Init(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == BLOCK_ATTN_RES_PREPARE_TPL_MIX) {
        AscendC::SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareMixTilingData, mixTilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareTensorApiBlazeKernel(blockRes, validBlocks, pseudoQuery, numerator,
                                                                     logitMax, expSum, userWorkspace, mixTilingData);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
