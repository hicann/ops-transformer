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
 * \brief Kernel entry for BlockAttnResPrepare on supported RegBase products.
 */

#include <type_traits>

#include "arch35/block_attn_res_prepare_apt_tiling_key.h"
#include "arch35/block_attn_res_prepare_mix.h"
#include "arch35/block_attn_res_prepare_vector.h"
#include "c_api/asc_simd.h"

template <uint32_t TEMPLATE_MODE>
__global__ __aicore__ void block_attn_res_prepare(GM_ADDR blockRes, GM_ADDR validBlocks, GM_ADDR pseudoQuery,
                                                  GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (TEMPLATE_MODE != BLOCK_ATTN_RES_PREPARE_MIX_TPL) {
        // Static tiling is local; only prefetch GM-backed tiling.
        if constexpr (std::is_same_v<__tiling_data_ptr__ optiling::BlockAttnResPrepareTilingData*,
                                     const __gm__ optiling::BlockAttnResPrepareTilingData*>) {
            asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(tiling), 0);
        }
    }
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(optiling::BlockAttnResPrepareTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 2)", optiling::BlockAttnResPrepareMixTilingData);
    if constexpr (TEMPLATE_MODE == BLOCK_ATTN_RES_PREPARE_ONLY_VECTOR_TPL ||
                  TEMPLATE_MODE == BLOCK_ATTN_RES_PREPARE_SINGLE_S_CACHE_TPL) {
        GET_TILING_DATA_PTR_WITH_STRUCT(optiling::BlockAttnResPrepareTilingData, tilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareVector<DTYPE_BLOCK_RES,
                                                       TEMPLATE_MODE == BLOCK_ATTN_RES_PREPARE_SINGLE_S_CACHE_TPL>
            op;
        op(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, tilingData);
    } else if constexpr (TEMPLATE_MODE == BLOCK_ATTN_RES_PREPARE_MIX_TPL) {
        AscendC::SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareMixTilingData, mixTilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareBlazeKernel(blockRes, validBlocks, pseudoQuery, numerator, logitMax,
                                                            expSum, userWorkspace, mixTilingData);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
