/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP8_SCALE_CACHE_LAYOUT_H
#define GBSA_MXFP8_SCALE_CACHE_LAYOUT_H
#include <cstdint>

#ifdef __CCE_AICORE__
#define GBSA_SCALE_INLINE __aicore__ inline
#else
#define GBSA_SCALE_INLINE inline
#endif

namespace GbsaMxScaleCache {
constexpr uint64_t CAPACITY = 128U * 1024U;
constexpr uint64_t UB_CAPACITY = 256U * 1024U;
constexpr uint64_t DESCRIPTOR_RING_SLOTS = 4U;
constexpr uint32_t GROUP_TOKENS = 128U;
constexpr uint32_t GROUPED_TILE_TOKENS = 512U;
constexpr uint32_t GROUPS_PER_TILE = GROUPED_TILE_TOKENS / GROUP_TOKENS;
constexpr uint32_t GROUPS_PER_AIV = GROUPS_PER_TILE / 2U;
constexpr uint32_t RAW_STAGES = 2U;
constexpr uint32_t GROUPED_RAW_STAGES = 3U;
constexpr uint32_t NZ_STAGES = 2U;
constexpr uint32_t RAW_ROW_BYTES = 32U;
constexpr uint32_t DIRECT_INDEX_COUNT_LIMIT = 8192U;
constexpr uint32_t UB_QUERY_ROW_ALIGNMENT = 16U;
constexpr uint32_t UB_STATS_ROWS = 64U;
constexpr uint32_t UB_COMPUTE_STAGES = 2U;
constexpr uint32_t UB_DELTA_STAGES = 3U;
constexpr uint32_t SCORE_ELEMENT_BYTES = sizeof(float);
constexpr uint32_t PROBABILITY_ELEMENT_BYTES = sizeof(uint8_t);
constexpr uint64_t L1_DATA_CAPACITY = 512U * 1024U - 16U * 1024U - UB_DELTA_STAGES * 4096U;

GBSA_SCALE_INLINE uint64_t Align(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

GBSA_SCALE_INLINE uint64_t BufferDim(uint64_t dim)
{
    // D192 and D256 share the D256 capacity tier. This is a reservation width,
    // never the active tensor width or a GM/scale address stride.
    return Align(dim, 128U);
}

// Head-group geometry is independent of the 128-token sparse descriptor groups.
GBSA_SCALE_INLINE uint32_t QueryTileRows(uint64_t group)
{
    return group <= 16U ? 16U : group <= 32U ? 32U : group <= 64U ? 64U : 128U;
}

GBSA_SCALE_INLINE uint32_t KvTileTokens(uint64_t group)
{
    return group <= 32U ? 512U : group <= 64U ? 256U : 128U;
}

GBSA_SCALE_INLINE uint32_t VectorRows(uint64_t qTile)
{
    // Each AIV reserves half the Q tile, rounded up to a 16-row capacity.
    return static_cast<uint32_t>(Align(qTile / 2U, UB_QUERY_ROW_ALIGNMENT));
}

GBSA_SCALE_INLINE uint64_t ScoreUbSlotBytes(uint64_t qTile, uint64_t tile)
{
    // All four supported M/KV tiers reserve 32 KiB per FP32 S slot per AIV.
    return VectorRows(qTile) * tile * SCORE_ELEMENT_BYTES;
}

GBSA_SCALE_INLINE uint32_t PvUbStages(uint64_t qTile)
{
    return qTile <= 64U ? UB_COMPUTE_STAGES : 1U;
}

GBSA_SCALE_INLINE uint64_t OutputUbSlotBytes(uint64_t dim, uint64_t qTile)
{
    return VectorRows(qTile) * BufferDim(dim) * sizeof(float);
}

GBSA_SCALE_INLINE uint64_t ValueScaleUbBytes(uint64_t dim)
{
    return BufferDim(dim) * sizeof(uint16_t);
}

GBSA_SCALE_INLINE uint64_t NzScaleUbSlotBytes(uint64_t dim)
{
    return Align(GROUP_TOKENS * (BufferDim(dim) / 32U), 256U);
}

GBSA_SCALE_INLINE uint32_t GroupsPerAiv(uint64_t tile)
{
    // A 128-token tile still gives AIV1 one empty notification to balance READY/FREE.
    return static_cast<uint32_t>((tile / GROUP_TOKENS + 1U) / 2U);
}

GBSA_SCALE_INLINE uint64_t RawTileTokens(uint64_t tile)
{
    return GroupsPerAiv(tile) * GROUP_TOKENS;
}

GBSA_SCALE_INLINE uint64_t NzTileTokens(uint64_t /* tile */)
{
    return GROUP_TOKENS;
}

GBSA_SCALE_INLINE uint64_t DescriptorGroupWords()
{
    // Repeated clipped tail blocks can contribute only one token per record.
    // Bound grouped records by tokens, not by the nominal sparse block size.
    return 8U + 2U * GROUP_TOKENS;
}

GBSA_SCALE_INLINE uint64_t DescriptorTileWords(uint64_t tile, uint64_t /* blockY */)
{
    // Keep a header for AIV1's empty group in the single-group tier.
    return 2U * GroupsPerAiv(tile) * DescriptorGroupWords();
}

GBSA_SCALE_INLINE uint64_t PipelineL1End(uint64_t dim, uint64_t qTile, uint64_t kvTile)
{
    // Q, three K groups, three full P tiles, and two full V tiles (all FP8).
    const uint64_t capacityDim = BufferDim(dim);
    return qTile * capacityDim + 3U * GROUP_TOKENS * capacityDim + UB_DELTA_STAGES * qTile * kvTile +
           2U * kvTile * capacityDim;
}

// Logical positioning is independent of an index cache in UB.
GBSA_SCALE_INLINE bool DirectIndexMapFits(uint64_t count)
{
    return count <= DIRECT_INDEX_COUNT_LIMIT;
}

GBSA_SCALE_INLINE uint32_t RawBlockRows(uint64_t blockY, bool headMajor)
{
    return headMajor ? static_cast<uint32_t>(blockY < 16U ? blockY : 16U) : 1U;
}

GBSA_SCALE_INLINE uint64_t RawCapacityBytes(uint64_t tokens)
{
    // Both layouts reserve 4 KiB per group and 8 KiB per AIV tile slot.
    // BNSD keeps its compact payload at the beginning of each group region.
    return tokens * RAW_ROW_BYTES;
}

GBSA_SCALE_INLINE uint64_t StreamBytes(uint64_t /* heads */, uint64_t dim, uint64_t tile, uint64_t blockY = 1U,
                                       bool headMajor = false)
{
    // BSND selects one head from each token and pads each row to 32 B.
    // BNSD keeps a contiguous microblock (at most 16 tokens) in one DMA block;
    // only that block's end is padded. This is the payload's padded footprint,
    // not the reserved capacity or the stride between groups/tile slots.
    const uint64_t rows = RawBlockRows(blockY, headMajor);
    return (tile + rows - 1U) / rows * Align(rows * (dim / 32U), RAW_ROW_BYTES);
}

GBSA_SCALE_INLINE bool StreamFits(uint64_t heads, uint64_t dim, uint64_t tile, uint64_t blockY = 1U,
                                  bool headMajor = false)
{
    const uint64_t capacity = RawCapacityBytes(tile);
    return heads != 0U && dim >= 32U && capacity <= CAPACITY &&
           StreamBytes(heads, dim, tile, blockY, headMajor) <= capacity;
}

// Raw is followed immediately by the independent per-group NZ outputs.
GBSA_SCALE_INLINE uint64_t PipelineAuxBytes(uint64_t dim, uint64_t /* tile */, uint32_t /* rawStages */ = RAW_STAGES)
{
    return NZ_STAGES * NzScaleUbSlotBytes(dim);
}

GBSA_SCALE_INLINE uint64_t PipelineCacheOffset(uint64_t dim, uint64_t tile, uint64_t qTile = 16U)
{
    const uint64_t rows = VectorRows(qTile);
    const uint64_t scoreBytes = ScoreUbSlotBytes(qTile, tile) * UB_COMPUTE_STAGES;
    const uint64_t probabilityBytes = rows * tile * PROBABILITY_ELEMENT_BYTES * UB_COMPUTE_STAGES;
    // The largest head group has one PV result buffer and one FP32 accumulator.
    const uint64_t outputBytes = OutputUbSlotBytes(dim, qTile) * (PvUbStages(qTile) + 1U);
    // Local/global maxima and sums, plus three online correction buffers.
    const uint64_t statisticsBytes = (4U + UB_DELTA_STAGES) * UB_STATS_ROWS * sizeof(float);
    const uint64_t rowScratchBytes = UB_STATS_ROWS * 32U;
    const uint64_t valueScaleBytes = ValueScaleUbBytes(dim);
    return Align(scoreBytes + probabilityBytes + outputBytes + statisticsBytes + rowScratchBytes + valueScaleBytes,
                 256U);
}

GBSA_SCALE_INLINE uint64_t PipelineUbEndForStages(uint64_t /* kv */, uint64_t /* heads */, uint64_t dim,
                                                  uint64_t blockY, uint64_t tile, uint32_t rawStages,
                                                  bool /* headMajor */ = false, uint64_t qTile = 16U)
{
    // Unsupported tile sizes must not accidentally pass the Host capacity check.
    if ((qTile != 16U && qTile != 32U && qTile != 64U && qTile != 128U) || tile != KvTileTokens(qTile)) {
        return UB_CAPACITY + 1U;
    }
    const uint64_t cacheOffset = PipelineCacheOffset(dim, tile, qTile);
    return cacheOffset + rawStages * RawCapacityBytes(RawTileTokens(tile)) + PipelineAuxBytes(dim, tile, rawStages) +
           DescriptorTileWords(tile, blockY) * sizeof(uint64_t);
}

GBSA_SCALE_INLINE uint32_t PipelineRawStages(uint64_t kv, uint64_t heads, uint64_t dim, uint64_t blockY, uint64_t tile,
                                             bool headMajor = false, uint64_t qTile = 16U)
{
    // Retain the grouped three-slot preference and its bounded two-slot fallback.
    return PipelineUbEndForStages(kv, heads, dim, blockY, tile, GROUPED_RAW_STAGES, headMajor, qTile) <= UB_CAPACITY ?
               GROUPED_RAW_STAGES :
               RAW_STAGES;
}

GBSA_SCALE_INLINE uint64_t PipelineUbEnd(uint64_t kv, uint64_t heads, uint64_t dim, uint64_t blockY, uint64_t tile,
                                         bool headMajor = false, uint64_t qTile = 16U)
{
    return PipelineUbEndForStages(kv, heads, dim, blockY, tile,
                                  PipelineRawStages(kv, heads, dim, blockY, tile, headMajor, qTile), headMajor, qTile);
}

} // namespace GbsaMxScaleCache
#undef GBSA_SCALE_INLINE
#endif
