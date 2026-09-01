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
 * \file ffn_to_attention_A5.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#if __has_include("../../ffn_to_attention/ffn_to_attention.h")
#include "../../ffn_to_attention/ffn_to_attention.h"
#else
#include "../../../ffn_to_attention/op_kernel/ffn_to_attention.h"
#endif
#include "../ffn_to_attention_v2_tiling.h"
#include "../ffn_to_attention_v2_tilling_key.h"
#include "../ffn_to_attention_urma.h"

using namespace AscendC;
using namespace FFNToAttentionImpl;
using namespace Mc2Tiling;

static_assert(sizeof(FFNToAttentionV2Info) == sizeof(FFNToAttentionInfo),
              "FFNToAttention V1/V2 tiling info layout size mismatch");
static_assert(sizeof(FFNToAttentionV2TilingData) == sizeof(FFNToAttentionTilingData),
              "FFNToAttention V1/V2 tiling data layout size mismatch");

/*
 * A5 tiling key fields:
 *   RankTableMode: whether attnRankTable is provided.
 *   ArchTag: A5.
 *   CommModeType: MTE or URMA.
 */
template <bool RankTableMode, uint8_t ArchTag>
__global__ __aicore__ void ffn_to_attention_v2(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds,
                                               GM_ADDR tokenIds, GM_ADDR expertOffsets, GM_ADDR actualTokenNum,
                                               GM_ADDR attnRankTable, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(FFNToAttentionV2TilingData);
    TPipe pipe;

    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(FFNToAttentionV2TilingData, tilingData, tilingGM);

        FFNToAttentionUrma<DTYPE_X, RankTableMode> op;
        op.Init(mc2Context, x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum, attnRankTable,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}
