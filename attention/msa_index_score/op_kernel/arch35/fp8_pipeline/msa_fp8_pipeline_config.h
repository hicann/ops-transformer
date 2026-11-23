/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <stdint.h>

namespace MsaNativeFp8Pipeline {
#ifndef MSA_K_L2_POLICY
#define MSA_K_L2_POLICY 0
#endif
constexpr int kKCachePolicy = MSA_K_L2_POLICY; // 0: allocate; 1: bypass on miss; 2: capacity heuristic
constexpr int64_t kL2Bytes = 128LL * 1024 * 1024;
constexpr int kArch = 3510;
constexpr int kHeads = 16;
constexpr int kDim = 128;
constexpr int kPageRows = 32;    // compressed rows, each summarizes four tokens
constexpr int kRowsPerScore = 4; // one block16 score
constexpr int kQueries = 8;      // same-request queries share the paged K loads
constexpr int kMaxM = kQueries * kHeads;
constexpr int kMaxN = 512;
#ifndef MSA_SCORE_PADDING
#define MSA_SCORE_PADDING 0
#endif
constexpr int kScorePadding = MSA_SCORE_PADDING; // FP32 elements per UB head row; not GM padding
constexpr int kPipelineMaxN = 256;
#ifndef MSA_SMALL_SCORE_STAGES
#define MSA_SMALL_SCORE_STAGES 2
#endif
constexpr int kSmallScoreStages = MSA_SMALL_SCORE_STAGES;
#ifndef MSA_UB_STAGES
#define MSA_UB_STAGES 3
#endif
constexpr int kLargeUbStages = MSA_UB_STAGES;
constexpr int kDmaBytes = 32;
constexpr int kScoreAlignment = 16;            // FP16 elements
constexpr float kScale = 0.12751743082459868f; // 128^-0.5 * log2(e), rounded once to FP32
constexpr float kNegative = -__builtin_inff();
// Byte addresses in distinct on-chip address spaces, not GM workspace.
constexpr int kQL1 = 0;
constexpr int kKL1 = kMaxM * kDim;
constexpr int kLargeScoreBytes = kLargeUbStages * (kMaxM / 2) * (kPipelineMaxN + kScorePadding) * sizeof(float);
constexpr int kSmallScoreBytes = kSmallScoreStages * (kMaxM / 2) * (128 + kScorePadding) * sizeof(float);
constexpr int kOutputUB = kLargeScoreBytes > kSmallScoreBytes ? kLargeScoreBytes : kSmallScoreBytes;
// DeepGEMM-Ascend style independent input/operand/epilogue queues. At N256:
// L1 = 16 KiB Q + 6*32 KiB K; L0B = 2*32 KiB; L0C = 2*128 KiB;
// Each AIV has up to 3*64 KiB FP32 scores and separate FP16 output slabs.
constexpr int kKeyStages = 6;
constexpr int kScoreStages = 2;
#ifndef MSA_OUTPUT_STAGES
#define MSA_OUTPUT_STAGES 2
#endif
constexpr int kOutputStages = MSA_OUTPUT_STAGES;
#ifndef MSA_FLUSH_TILES
#define MSA_FLUSH_TILES 8
#endif
#ifndef MSA_CONTIGUOUS_TILES
#define MSA_CONTIGUOUS_TILES 1
#endif
constexpr int kFlushTiles = MSA_FLUSH_TILES;
constexpr bool kContiguousTiles = MSA_CONTIGUOUS_TILES;
#ifndef MSA_COALESCE_PAGES
#define MSA_COALESCE_PAGES 1
#endif
constexpr bool kCoalescePages = MSA_COALESCE_PAGES;
constexpr int kOutputStageBytes = (kQueries / 2) * (kPipelineMaxN / 4) * sizeof(uint16_t) * kFlushTiles;
constexpr int kQueryEvent = kKeyStages;
constexpr int kMadQueries = 2;
constexpr int kMadRows = kMadQueries * kHeads;
constexpr int kMadStages = 4; // whole M32xN512 FP32 tiles, round-robin over two AIVs
constexpr int kTableUB = kOutputUB + kOutputStages * kOutputStageBytes;
constexpr int kTableCacheBytes = 16 * 1024;
constexpr int kReversalUB = kTableUB + kTableCacheBytes;
constexpr int kReversalBytes = kTableCacheBytes / 8;
static_assert(kReversalUB + kReversalBytes <= 248 * 1024);
} // namespace MsaNativeFp8Pipeline

// Internal adaptation of the customer descriptor, not a new public ABI.
// q_prefix/k_lengths bind existing MsaIndexScore device metadata.
// Legacy descriptor fields are retained for upstream source correspondence.
// Inputs' head/dimension axes are contiguous; row/page/table strides are explicit.
struct MsaFp8PipelineArgs {
    uint64_t q, k, table, token_batch, positions, visible, out, bs;
    int64_t tokens, batches, pages, table_columns, scores;
    int64_t q_stride, k_stride, table_stride, out_stride;
    int64_t token_stride, position_stride, visible_stride;
    int64_t query_tile, summary_tile;
    uint64_t q_prefix, k_lengths;
    int64_t causal;
};
