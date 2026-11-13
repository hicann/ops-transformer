/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_ARCH35_LAYOUT_UTILS_HPP
#define GBSA_ARCH35_LAYOUT_UTILS_HPP

#include <cstdint>
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#define GBSA_LAYOUT_INLINE __aicore__ inline
#else
// CPU UT may instantiate device helpers without __aicore__.
#define GBSA_LAYOUT_INLINE inline
#endif

namespace GbsaKernelArch35 {

enum class Format {
    TND = 0,
    BNSD = 1,
    BSND = 2
};

// Sorted valid-prefix indices. FD ranges use virtual block coordinates: explicit
// blocks [0, explicitCount), optionally followed by one implicit residual block.
template <bool hasResidualBlock, class Indices>
GBSA_LAYOUT_INLINE void CalcSparseKvSize(Indices& indices, uint32_t explicitCount, uint32_t rangeBegin,
                                         uint32_t rangeEnd, uint32_t blockShapeY, uint32_t visibleEnd,
                                         uint32_t& validTokens)
{
    if constexpr (hasResidualBlock) {
        const uint32_t explicitBegin = rangeBegin < explicitCount ? rangeBegin : explicitCount;
        const uint32_t explicitEnd = rangeEnd < explicitCount ? rangeEnd : explicitCount;
        validTokens = (explicitEnd - explicitBegin) * blockShapeY;
        const uint32_t residualSize = visibleEnd % blockShapeY;
        if (residualSize != 0U && rangeBegin <= explicitCount && rangeEnd > explicitCount) {
            validTokens += residualSize;
        }
    } else {
        const uint32_t count = rangeEnd - rangeBegin;
        validTokens = count * blockShapeY;
        if (count != 0U) {
            const uint64_t lastStart = static_cast<uint64_t>(indices.GetValue(rangeEnd - 1U)) * blockShapeY;
            const uint32_t remaining = static_cast<uint32_t>(visibleEnd - lastStart);
            const uint32_t lastSize = remaining < blockShapeY ? remaining : blockShapeY;
            validTokens -= blockShapeY - lastSize;
        }
    }
}

template <bool hasResidualBlock, class Indices>
GBSA_LAYOUT_INLINE int32_t GetSparseBlockId(Indices& indices, uint32_t localBlockIdx, uint32_t localExplicitCount,
                                            uint32_t residualBlockId)
{
    if constexpr (hasResidualBlock) {
        if (localBlockIdx == localExplicitCount) {
            return static_cast<int32_t>(residualBlockId);
        }
    }
    return indices.GetValue(localBlockIdx);
}

// Four slots cover QK/PV prelaunch and one tile of address lookahead.
// Each group: [copy count, 7 reserved words], followed by plain uint64 entries.
// Meta packs (sourceStride << 32) | (length << 1) | (ndNum - 1).
// Dense: [element offset, meta], retaining the original four-word capacity.
// PA: [K element offset, V element offset, K meta, V meta].
constexpr uint32_t KV_OFFSET_STAGES = 4U;
constexpr uint32_t KV_OFFSET_FLAG_BASE = 7U;
constexpr uint32_t KV_OFFSET_FIRST_READY_FLAG_BASE = KV_OFFSET_FLAG_BASE + KV_OFFSET_STAGES;

template <bool isPaged>
GBSA_LAYOUT_INLINE uint32_t CalcKvOffsetGroupWords(uint32_t l1TileN, uint32_t blockY)
{
    uint32_t blocks = (l1TileN + 2U * blockY - 2U) / blockY;
    if constexpr (isPaged) {
        blocks = blocks * 2U < l1TileN ? blocks * 2U : l1TileN;
    }
    constexpr uint32_t recordWords = 4U;
    return (8U + blocks * recordWords + 7U) / 8U * 8U;
}

// BSA-style tensor strides. Packed GQA traverses heads along the GEMM M axis,
// so its Q/O row stride is HeadStride, not TokenStride.
template <Format format>
struct GbsaLayoutHelper {
    GBSA_LAYOUT_INLINE static uint64_t TokenStride(uint32_t heads, uint32_t dim)
    {
        if constexpr (format == Format::BNSD) {
            return dim;
        }
        return static_cast<uint64_t>(heads) * dim;
    }

    GBSA_LAYOUT_INLINE static uint64_t HeadStride(uint32_t seqCapacity, uint32_t dim)
    {
        if constexpr (format == Format::BNSD) {
            return static_cast<uint64_t>(seqCapacity) * dim;
        }
        return dim;
    }

    GBSA_LAYOUT_INLINE static uint64_t Offset(uint64_t batchBase, uint32_t token, uint32_t head, uint32_t heads,
                                              uint32_t dim, uint32_t seqCapacity = 0)
    {
        return batchBase + static_cast<uint64_t>(token) * TokenStride(heads, dim) +
               static_cast<uint64_t>(head) * HeadStride(seqCapacity, dim);
    }
};

struct BatchSeqInfo {
    // Flattened storage-token base: cu[b] for TND, b * seqCapacity for batched layouts.
    // This is not yet an element offset; each tensor layout supplies its own strides.
    uint64_t storageStart;
    uint32_t storageLen;
    uint32_t actualLen;
};

template <Format format, class CuTensor, class UsedTensor>
GBSA_LAYOUT_INLINE void ReadBatchSeqInfo(CuTensor& cu, UsedTensor& used, bool hasUsed, uint32_t batch,
                                         BatchSeqInfo& seq, uint32_t seqCapacity = 0)
{
    if constexpr (format == Format::TND) {
        const int64_t start = cu.GetValue(batch);
        const int64_t end = cu.GetValue(static_cast<int64_t>(batch) + 1);
        seq.storageStart = static_cast<uint64_t>(start);
        seq.storageLen = static_cast<uint32_t>(end - start);
        seq.actualLen = hasUsed ? static_cast<uint32_t>(used.GetValue(batch)) : seq.storageLen;
    } else {
        seq.storageStart = static_cast<uint64_t>(batch) * seqCapacity;
        seq.storageLen = seqCapacity;
        seq.actualLen = static_cast<uint32_t>(used.GetValue(batch));
    }
}

template <class UsedTensor>
GBSA_LAYOUT_INLINE void DecodeBatchedQToken(uint32_t qToken, uint32_t batchCount, uint32_t seqCapacity,
                                            UsedTensor& used, uint32_t& batchIdx, uint32_t& tokenInBatch,
                                            BatchSeqInfo& seq)
{
    uint32_t actualPrefix = 0U;
    for (uint32_t b = 0U; b < batchCount; ++b) {
        seq.storageStart = static_cast<uint64_t>(b) * seqCapacity;
        seq.storageLen = seqCapacity;
        seq.actualLen = static_cast<uint32_t>(used.GetValue(b));
        if (qToken < actualPrefix + seq.actualLen) {
            batchIdx = b;
            tokenInBatch = qToken - actualPrefix;
            return;
        }
        actualPrefix += seq.actualLen;
    }
}

// Q/O addressing uses the same batch-local token/head coordinates for every layout.
template <Format format>
GBSA_LAYOUT_INLINE uint64_t GetTensorOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head, uint32_t heads,
                                            uint32_t dim)
{
    return GbsaLayoutHelper<format>::Offset(seq.storageStart * heads * dim, token, head, heads, dim, seq.storageLen);
}

template <Format qFormat>
GBSA_LAYOUT_INLINE uint64_t GetLseOffset(const BatchSeqInfo& seq, uint32_t token, uint32_t head, uint32_t heads)
{
    // GBSA API: TND -> [T,N,1]; both BSND and BNSD -> [B,N,S,1].
    constexpr Format lseLayout = qFormat == Format::TND ? Format::TND : Format::BNSD;
    return GetTensorOffset<lseLayout>(seq, token, head, heads, 1);
}

template <Format kvFormat, bool isPaged>
GBSA_LAYOUT_INLINE uint64_t KvBlockOffset(uint32_t blockId, const BatchSeqInfo& seq, uint32_t head, uint32_t heads,
                                          uint32_t dim, uint32_t blockShapeY, uint64_t pageStride)
{
    if constexpr (isPaged) {
        return GbsaLayoutHelper<kvFormat>::Offset(static_cast<uint64_t>(blockId) * pageStride, 0, head, heads, dim,
                                                  blockShapeY);
    } else {
        return GetTensorOffset<kvFormat>(seq, 0, head, heads, dim) +
               static_cast<uint64_t>(blockId) * blockShapeY * GbsaLayoutHelper<kvFormat>::TokenStride(heads, dim);
    }
}

// Union (maskType=3) device UB packing. Host tiling must not include this header;
// it uses a separate tile-cap policy instead of recomputing this layout.
constexpr uint32_t USABLE_UB_BYTES = 248U * 1024U;
constexpr uint32_t STATS_ROWS = 64U;
constexpr uint32_t EXACT_MASK_STAGES = 4U;

struct UbLayout {
    uint32_t qRows, kvRows, dimRows, loBuffers;
    uint32_t s, p, lo, go, lm, gm, dm, ll, gl, lse, descriptors;
    uint32_t descriptorBuffers, exactMask, partialMax, end;
};

GBSA_LAYOUT_INLINE constexpr uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

GBSA_LAYOUT_INLINE constexpr UbLayout MakeUbLayout(uint32_t qTile, uint32_t kvTile, uint32_t dim, uint32_t l1KvTile,
                                                   uint32_t blockY, bool paged, bool fp8)
{
    UbLayout b{};
    b.qRows = AlignUp(qTile / 2U, 16U);
    b.kvRows = AlignUp(kvTile, 32U);
    b.dimRows = AlignUp(dim, 32U);
    b.loBuffers = (qTile > 64U && (dim > 128U || (fp8 && kvTile >= 256U))) ? 1U : 2U;
    b.s = 0U;
    b.p = b.qRows * b.kvRows * sizeof(float) * 2U;
    const uint32_t pBytes = fp8 ? (b.kvRows / 4U + 1U) * 256U * 2U : (b.kvRows / 2U + 1U) * 128U * 2U * 2U;
    b.lo = b.p + pBytes;
    b.go = b.lo + b.qRows * b.dimRows * sizeof(float) * b.loBuffers;
    b.lm = b.go + b.qRows * b.dimRows * sizeof(float);
    b.gm = b.lm + STATS_ROWS * sizeof(float);
    b.dm = b.gm + STATS_ROWS * sizeof(float);
    b.ll = b.dm + STATS_ROWS * sizeof(float) * 3U;
    b.gl = b.ll + STATS_ROWS * sizeof(float);
    b.lse = b.gl + STATS_ROWS * sizeof(float);
    b.descriptors = b.lse + STATS_ROWS * 32U;
    uint32_t records = (l1KvTile + 2U * blockY - 2U) / blockY;
    if (paged)
        records = records * 2U < l1KvTile ? records * 2U : l1KvTile;
    const uint32_t descriptorBytes = AlignUp(8U + records * 4U, 8U) * sizeof(uint64_t);
    const uint32_t maskBytes = EXACT_MASK_STAGES * b.kvRows * sizeof(int32_t);
    const uint32_t scratchBytes = STATS_ROWS * sizeof(float) + (fp8 ? 256U : 0U);
    b.descriptorBuffers = b.descriptors + 2U * descriptorBytes + maskBytes + scratchBytes <= USABLE_UB_BYTES ? 2U : 1U;
    b.exactMask = b.descriptors + b.descriptorBuffers * descriptorBytes;
    b.partialMax = b.exactMask + maskBytes;
    b.end = b.partialMax + scratchBytes;
    return b;
}

} // namespace GbsaKernelArch35

#endif // GBSA_ARCH35_LAYOUT_UTILS_HPP
