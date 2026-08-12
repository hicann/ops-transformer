/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_COMMON_HPP
#define MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_COMMON_HPP

#include "kernel_operator.h"

namespace MinimaxSaSplitKv {

struct MinimaxSparseAttentionSplitKvTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t groupSize;
    uint32_t embeddingSize;
    uint32_t blockSize;
    uint32_t topK;
    uint32_t totalQTokens;
    uint32_t numKvBlocks;
    uint32_t maxBlocksPerBatch;
    uint32_t k2qNnzUpperBound;
    uint32_t totalTaskNumP1;
    uint32_t totalTaskNumP2;
    float scaleValue;
    uint32_t innerPrecise;
    uint64_t accumOutSize;
    uint64_t lseStatSize;
    uint64_t workSpaceSize;
    uint64_t tilingKey;
};

}  // namespace MinimaxSaSplitKv

#endif  // MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_COMMON_HPP
