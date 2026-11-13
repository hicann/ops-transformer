/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON_HPP
#define KERNEL_COMMON_HPP

#include "kernel_operator.h"

namespace GenericBlockSparseAttn {

struct GbsaBaseTiling {
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
};

struct GbsaMmPhaseL1Tiling {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;
};

// Kernel-side mirror of GenericBlockSparseAttentionTilingData (host tiling).
// Field order must match BEGIN_TILING_DATA_DEF in generic_block_sparse_attention_tiling.h.
struct GenericBlockSparseAttentionTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t blockShapeX; // Q sparse block size (default 1)
    uint32_t blockShapeY; // KV sparse block size (default 128)
    uint32_t blockSize;   // Paged KV cache block size, parsed from KV tensor shape
    uint32_t topK;        // sparseBlockIdx dim-3 (KVb): selected KV blocks per Q block
    uint32_t qBlockNum;   // sparseBlockIdx dim-2 (Qb): total Q blocks
    uint32_t maxBlocksPerBatch;
    float scaleValue;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t updateSize;
    uint64_t workSpaceSize;
    uint64_t tilingKey;
    uint32_t groupSize;
    // PA_BBND page base stride in elements (dim0). Allows first-axis non-contiguous KV cache.
    uint64_t kStride0;
    uint64_t vStride0;
    uint32_t fdStaticEnabled;
    uint32_t fdLseSubStride;
    uint32_t fdPartialCapacity;
    uint64_t fdPartialLseOffset;
    uint64_t fdPartialOOffset;
    uint32_t residualBlockMode;
    uint32_t isConsistentTopk;
    uint64_t totalQTokens;
    uint64_t totalKvTokens; // PA: physical page token capacity; Dense: Tkv.
    GbsaBaseTiling baseTileInfo;
    GbsaMmPhaseL1Tiling mmPhaseL1TileInfo;
    // Nonquantized: descriptor workspace base and words per tile.
    // MXFP8 packed: immutable batch/head cache base (0 disables) and words per K/V tensor.
    uint64_t kvOffsetInfoOffset;
    uint32_t kvOffsetTileWords;
    uint32_t qSeqCapacity;
    uint32_t kvSeqCapacity;
    // Mask mode consumed by arch22: 0 = none, 1 = built-in causal.
    uint32_t maskType;
    // Arch22 layout strides: 0=TND/PA_BBND, 1=BSND, 2=BNSD, 3=PA_BNBD.
    uint32_t qLayout;
    uint32_t kvLayout;
    // Union packed O/LSE scratch when multi-token GQA rows are non-contiguous in GM.
    uint64_t unionIoPackOffset;
    uint32_t unionIoPackORowStride;
};

} // namespace GenericBlockSparseAttn

#endif // KERNEL_COMMON_HPP
