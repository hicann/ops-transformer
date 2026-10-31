/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STA_TILING_DATA_H
#define STA_TILING_DATA_H

#include <cstdint>

namespace optiling {

// Kernel-side view of SlidingTileAttentionTilingData. Every field is 64-bit so the layout written by the host
// TILING_DATA_FIELD_DEF serializer has no padding; field order must match sliding_tile_attention_tiling.h.
struct SlidingTileAttentionTilingData {
    uint64_t batch = 0;
    uint64_t numHeads = 0;
    uint64_t embeddingSize = 0;
    uint64_t kvBlockSize = 0;
    uint64_t firstBatchTaskNum = 0;
    uint64_t totalTaskNum = 0;
    uint64_t blockShapeX = 0;
    uint64_t blockShapeY = 0;
    uint64_t seqLen = 0;
    uint64_t validKvSeqLen = 0;
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t updateSize = 0;
    uint64_t selectIdxSize = 0;
    uint64_t scaleValueBits = 0;
    uint64_t sumColumn = 0;             // non-zero: first padded value column; the kernel fills the padding with ones
    uint64_t totalRows = 0;             // B * N * S
    uint64_t prepassRowsPerSubCore = 0; // rows of the query/value preparation per vector sub-core
    uint64_t stage2Delay = 0;           // KV stacks between a stack's stage 1 and its stage 2
    uint64_t warmQueries = 0;           // 1: the vector cores read the queries of upcoming tasks into L2
    uint64_t tilesT = 0;                // tile grid of the image tokens
    uint64_t tilesH = 0;
    uint64_t tilesW = 0;
    uint64_t imageBlocks = 0;        // tilesT * tilesH * tilesW
    uint64_t blockNum = 0;           // 384-token blocks of the sequence (image blocks + at most one text block)
    uint64_t textSelect = 0;         // 1: image queries also attend the text block (textLength > 0)
    uint64_t selectCap = 0;          // entries reserved for the KV block list of an image block (C0 multiple)
    uint64_t listRowNum = 0;         // image block lists: heads * imageBlocks
    uint64_t listRowsPerSubCore = 0; // lists written by each vector sub-core
    uint64_t listSubCoreNum = 0;     // vector sub-cores that write lists
};

} // namespace optiling

#endif // STA_TILING_DATA_H
