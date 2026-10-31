/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_TILING_H
#define SLIDING_TILE_ATTENTION_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

// Serialized tiling data. Field order and widths must match op_kernel/sta_tiling_data.h.
BEGIN_TILING_DATA_DEF(SlidingTileAttentionTilingData)
TILING_DATA_FIELD_DEF(uint64_t, batch)
TILING_DATA_FIELD_DEF(uint64_t, numHeads)
TILING_DATA_FIELD_DEF(uint64_t, embeddingSize)
TILING_DATA_FIELD_DEF(uint64_t, kvBlockSize)
TILING_DATA_FIELD_DEF(uint64_t, firstBatchTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, totalTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeX)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeY)
TILING_DATA_FIELD_DEF(uint64_t, seqLen)
TILING_DATA_FIELD_DEF(uint64_t, validKvSeqLen)
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize)
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize)
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize)
TILING_DATA_FIELD_DEF(uint64_t, updateSize)
TILING_DATA_FIELD_DEF(uint64_t, selectIdxSize)
TILING_DATA_FIELD_DEF(uint64_t, scaleValueBits)
TILING_DATA_FIELD_DEF(uint64_t, sumColumn)
TILING_DATA_FIELD_DEF(uint64_t, totalRows)
TILING_DATA_FIELD_DEF(uint64_t, prepassRowsPerSubCore)
TILING_DATA_FIELD_DEF(uint64_t, stage2Delay)
TILING_DATA_FIELD_DEF(uint64_t, warmQueries)
TILING_DATA_FIELD_DEF(uint64_t, tilesT)
TILING_DATA_FIELD_DEF(uint64_t, tilesH)
TILING_DATA_FIELD_DEF(uint64_t, tilesW)
TILING_DATA_FIELD_DEF(uint64_t, imageBlocks)
TILING_DATA_FIELD_DEF(uint64_t, blockNum)
TILING_DATA_FIELD_DEF(uint64_t, textSelect)
TILING_DATA_FIELD_DEF(uint64_t, selectCap)
TILING_DATA_FIELD_DEF(uint64_t, listRowNum)
TILING_DATA_FIELD_DEF(uint64_t, listRowsPerSubCore)
TILING_DATA_FIELD_DEF(uint64_t, listSubCoreNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SlidingTileAttention, SlidingTileAttentionTilingData)

struct SlidingTileAttentionCompileInfo {
    uint32_t aicNum = 1;
    uint32_t aivNum = 1;
    uint64_t ubSize = 0;
    platform_ascendc::SocVersion socVersion;
};

} // namespace optiling

#endif // SLIDING_TILE_ATTENTION_TILING_H
