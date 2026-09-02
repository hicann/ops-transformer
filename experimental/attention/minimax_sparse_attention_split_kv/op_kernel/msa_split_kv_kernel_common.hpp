/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MINIMAX_SPARSE_ATTENTION_SPLIT_KV_MSA_SPLIT_KV_KERNEL_COMMON_HPP
#define MINIMAX_SPARSE_ATTENTION_SPLIT_KV_MSA_SPLIT_KV_KERNEL_COMMON_HPP

#include "kernel_operator.h"

namespace MinimaxSaSplitKv {

struct MinimaxSparseAttentionSplitKvTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t groupSize;
    uint32_t embeddingSize;
    uint32_t blockSize;
    uint64_t queryTokenStride; // Q token stride in elements (A2)
    uint64_t keyBlockStride;   // K cache physical-block stride in elements (A2)
    uint64_t valueBlockStride; // V cache physical-block stride in elements (A2)
    uint32_t topK;
    uint32_t totalQTokens;
    uint32_t numKvBlocks;
    uint32_t maxBlocksPerBatch;
    uint32_t k2qNnzUpperBound;
    uint32_t totalTaskNumP1;
    uint32_t totalTaskNumP2;
    float scaleValue;
    uint32_t innerPrecise;
    uint32_t softmaxLseFlag;
    uint64_t accumOutSize;
    uint64_t lseStatSize;
    uint64_t workSpaceSize;
    uint64_t gmSWorkspaceSize; // A2 GM S staging workspace per-core bytes (A5=0)
    uint64_t gmPWorkspaceSize; // A2 GM P staging workspace per-core bytes (A5=0)
    uint64_t tilingKey;
    uint32_t isKvContinuous;   // A2: KV contiguous vs paged
    uint64_t keyTokenStride;   // A2: K token stride for contiguous KV
    uint64_t valueTokenStride; // A2: V token stride for contiguous KV
    uint32_t isPageAttention;  // master: paged attention flag
    uint32_t layoutType;       // master: TND/BNSD/BSND layout
    uint32_t qSeqLen;          // master: per-batch Q seq len (kv gather q)
    uint32_t kvSeqLen;         // master: per-batch KV seq len (kv gather q)
};

constexpr uint32_t LAYOUT_TND = 0;
constexpr uint32_t LAYOUT_BNSD = 1;
constexpr uint32_t LAYOUT_BSND = 2;

} // namespace MinimaxSaSplitKv

#endif // MINIMAX_SPARSE_ATTENTION_SPLIT_KV_MSA_SPLIT_KV_KERNEL_COMMON_HPP
