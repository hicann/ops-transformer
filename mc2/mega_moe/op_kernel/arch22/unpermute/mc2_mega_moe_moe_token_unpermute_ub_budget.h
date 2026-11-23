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
 * \file mc2_mega_moe_moe_token_unpermute_ub_budget.h
 * \brief Pure-integer UB budget for the unpermute hidden-dimension tile.
 *
 * Plain C++ on purpose: no AscendC headers, no intrinsics, so the host stub in
 * tests/st can include it and sweep the parameter space without CANN.
 */

#ifndef MOE_TOKEN_UNPERMUTE_UB_BUDGET_MC2
#define MOE_TOKEN_UNPERMUTE_UB_BUDGET_MC2

#include <cstdint>

// AscendC device callers require device-qualified helpers. Keep host-only
// budget tests independent of AscendC headers and compiler keywords.
#if defined(__CCE_AICORE__)
#define MEGA_MOE_UNPERMUTE_BUDGET_INLINE __aicore__ inline
#else
#define MEGA_MOE_UNPERMUTE_BUDGET_INLINE inline
#endif

namespace MoeTokenUnpermuteUbBudget {

// Unified buffer size of one AIV on ascend910_93, mirroring
// template_linear_algebra/arch/tla_arch_arch.hpp:19 (ArchTag::UB_SIZE).
constexpr int64_t UB_TOTAL_BYTES = 192 * 1024;

// The A3 kernel parks three caches at the TOP of UB while unpermute's TPipe
// allocates from the bottom, so the top slice must be excluded here. See
// mega_moe_kernel_a3.hpp InitRankMask() and InitRouteCache(). unpermute never
// receives EP or expertPerRank, so this is a static upper bound; with EP <= 128
// and EP * expertPerRank <= 1024:
//   rankMask   = AlignUp(EP * 4, 32)                   <=   512 B
//   tpeCache   = AlignUp(EP * AlignUp(epr, 8) * 4, 32) <=  7680 B
//   routeCache = AlignUp(EP * epr * 4, 32)             <=  4096 B
// i.e. under 12 KB in total, so 16 KB keeps headroom without plumbing EP and
// expertPerRank down into unpermute.
//
// Scope: A3 only. mega_moe_kernel_a2.hpp (and its a2_bf16_w4a8 twin) lay out
// 3 * routeCache + tpeCache instead, which reaches 19520 B, and they call this
// kernel with sharedExpertNum as well. ClampHiddenTiling therefore keeps the
// legacy 1024 limit unless the caller explicitly opts into the A3 budget.
constexpr int64_t UB_TOP_RESERVE_BYTES = 16 * 1024;

// DataCopyPad on the hidden dimension wants 512-byte tiles, and TPipe hands out
// 32-byte granules. A 512-aligned tile therefore has no allocation slack.
constexpr int64_t TILE_ALIGN_BYTES = 512;
constexpr int64_t TPIPE_ALIGN_BYTES = 32;

MEGA_MOE_UNPERMUTE_BUDGET_INLINE int64_t AlignUpTo(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}

MEGA_MOE_UNPERMUTE_BUDGET_INLINE int64_t AlignDownTo(int64_t value, int64_t align)
{
    return value / align * align;
}

// Bytes consumed per hidden element, i.e. the buffers InitLocalBuffers() sizes
// by hidden_splited_length_align512. Only the non-float T1 path is modelled;
// that is the only path that tiles the hidden dimension.
MEGA_MOE_UNPERMUTE_BUDGET_INLINE int64_t HiddenScaledBytesPerElement(int64_t bufferNum, int64_t t1Bytes, bool hasShared)
{
    // tokens_inque (bufferNum deep) + outque
    int64_t bytes = bufferNum * t1Bytes + t1Bytes;
    // temp_buffer0 + temp_buffer1, both float
    bytes += 4 + 4;
    if (hasShared) {
        // shared_sum_buffer (float) + branch_round_buffer (T1)
        bytes += 4 + t1Bytes;
    }
    return bytes;
}

// Bytes that do not scale with the hidden tile: the index/prob queues, which are
// sized by block_splited_length, plus the fixed 256-byte pad on temp_buffer0.
// Each InitBuffer call is rounded up to TPipe's 32-byte granularity.
MEGA_MOE_UNPERMUTE_BUDGET_INLINE int64_t HiddenInvariantBytes(int64_t blockSplitedLength, int64_t t2Bytes,
                                                              int64_t t3Bytes, bool hasProbs, bool probsNeedCast)
{
    int64_t bytes = AlignUpTo(blockSplitedLength * t2Bytes, TPIPE_ALIGN_BYTES);
    if (hasProbs) {
        bytes += AlignUpTo(blockSplitedLength * t3Bytes, TPIPE_ALIGN_BYTES);
        if (probsNeedCast) {
            // temp_buffer2 exists only when T3 is not float.
            bytes += AlignUpTo(blockSplitedLength * 4, TPIPE_ALIGN_BYTES);
        }
    }
    bytes += AlignUpTo(256, TPIPE_ALIGN_BYTES);
    return bytes;
}

// Largest 512-aligned hidden tile that still fits the UB budget. Flooring to 512
// is what makes AlignUp512(min(hidden_size, tile)) <= tile hold, so a hidden_size
// that is not a multiple of 512 cannot overshoot either.
MEGA_MOE_UNPERMUTE_BUDGET_INLINE int64_t HiddenTileLimit(int64_t blockSplitedLength, int64_t bufferNum, int64_t t1Bytes,
                                                         int64_t t2Bytes, int64_t t3Bytes, bool hasProbs,
                                                         bool probsNeedCast, bool hasShared)
{
    const int64_t perElement = HiddenScaledBytesPerElement(bufferNum, t1Bytes, hasShared);
    const int64_t invariant = HiddenInvariantBytes(blockSplitedLength, t2Bytes, t3Bytes, hasProbs, probsNeedCast);
    const int64_t available = UB_TOTAL_BYTES - UB_TOP_RESERVE_BYTES - invariant;
    if (available <= 0) {
        return TILE_ALIGN_BYTES;
    }
    const int64_t tile = AlignDownTo(available / perElement, TILE_ALIGN_BYTES);
    // One 512-element tile is the smallest unit the hidden loop can carry; a
    // config that cannot afford even that is out of range for this kernel and
    // would already have overflowed UB before this clamp existed.
    return tile < TILE_ALIGN_BYTES ? TILE_ALIGN_BYTES : tile;
}

// The three tiling fields the hidden loop consumes, kept together so the clamp
// cannot leave them inconsistent. Invariant: num * length + remain == hidden_size.
struct HiddenTiling {
    int64_t length;
    int64_t num;
    int64_t remain;
};

// A2 and unspecified callers retain the legacy 1024 clamp. Only A3 callers may
// opt into the budget above: its reserve does not cover the A2 cache layout.
// A no-op when the tile from tiling already fits the selected limit.
template <bool UseA3UbBudget = false>
MEGA_MOE_UNPERMUTE_BUDGET_INLINE HiddenTiling ClampHiddenTiling(HiddenTiling fromTiling, int64_t hiddenSize,
                                                                int64_t blockSplitedLength, int64_t bufferNum,
                                                                int64_t t1Bytes, int64_t t2Bytes, int64_t t3Bytes,
                                                                bool hasProbs, bool probsNeedCast, bool hasShared)
{
    int64_t limit = 1024;
    if constexpr (UseA3UbBudget) {
        limit = HiddenTileLimit(blockSplitedLength, bufferNum, t1Bytes, t2Bytes, t3Bytes, hasProbs, probsNeedCast,
                                hasShared);
    }
    if (fromTiling.length <= limit) {
        return fromTiling;
    }
    HiddenTiling clamped;
    clamped.length = limit;
    clamped.num = hiddenSize / limit;
    clamped.remain = hiddenSize % limit;
    return clamped;
}

// Total UB bytes InitLocalBuffers() would allocate for a given hidden tile.
// Exposed so the host stub can assert the budget directly instead of trusting
// the inverse computation above.
MEGA_MOE_UNPERMUTE_BUDGET_INLINE int64_t TotalUbBytes(int64_t hiddenTile, int64_t blockSplitedLength, int64_t bufferNum,
                                                      int64_t t1Bytes, int64_t t2Bytes, int64_t t3Bytes, bool hasProbs,
                                                      bool probsNeedCast, bool hasShared)
{
    const int64_t aligned = AlignUpTo(hiddenTile, TILE_ALIGN_BYTES);
    return aligned * HiddenScaledBytesPerElement(bufferNum, t1Bytes, hasShared) +
           HiddenInvariantBytes(blockSplitedLength, t2Bytes, t3Bytes, hasProbs, probsNeedCast);
}

} // namespace MoeTokenUnpermuteUbBudget

#undef MEGA_MOE_UNPERMUTE_BUDGET_INLINE

#endif
