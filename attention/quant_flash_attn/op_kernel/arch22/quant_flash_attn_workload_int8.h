/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_WORKLOAD_INT8_H_
#define QUANT_FLASH_ATTN_WORKLOAD_INT8_H_
#include <cstdint>
#include "quant_flash_attn_tiling_data_int8.h"
#ifndef QFA_WORKLOAD_INLINE
#define QFA_WORKLOAD_INLINE __aicore__ inline
#define QFA_WORKLOAD_LOCAL_INLINE
#endif
namespace BaseApi::Int8GqaTnd {
struct WindowRange {
    uint32_t begin;
    uint32_t end;
};
QFA_WORKLOAD_INLINE uint32_t QfaWindowClip(int64_t x, uint32_t limit)
{
    return x <= 0 ? 0U : x >= static_cast<int64_t>(limit) ? limit : static_cast<uint32_t>(x);
}
// Inclusive left/right distances around the bottom-right aligned query. Return a half-open KV interval; -1 on
// either side means unbounded.
QFA_WORKLOAD_INLINE WindowRange QfaWindowRow(uint32_t token, uint32_t sq, uint32_t sk, int32_t left, int32_t right)
{
    const int64_t center = static_cast<int64_t>(token) + sk - sq;
    const uint32_t begin = left < 0 ? 0U : QfaWindowClip(center - left, sk);
    const uint32_t end = right < 0 ? sk : QfaWindowClip(center + right + 1, sk);
    return {begin, end < begin ? begin : end};
}
QFA_WORKLOAD_INLINE WindowRange QfaWindowTile(WindowRange row, uint32_t start, uint32_t count)
{
    const uint32_t begin = QfaWindowClip(static_cast<int64_t>(row.begin) - start, count);
    const uint32_t end = QfaWindowClip(static_cast<int64_t>(row.end) - start, count);
    return {begin, end < begin ? begin : end};
}

// The host task count and kernel decoder must use the same head-aligned tiling rule.
template <typename SequenceLength>
QFA_WORKLOAD_INLINE bool QfaUseHeadAlignedTiles(SequenceLength seqLenQ, SequenceLength seqLenKv, uint32_t headDim,
                                                uint32_t queryTileRows)
{
    constexpr uint32_t maxShortSequenceTokens = 512U;
    constexpr uint32_t minHeadDimWithoutTailLimit = 64U;
    constexpr uint32_t tailTileDivisor = 4U; // At most one quarter of a tile may be a tail for smaller D.
    return seqLenQ > maxShortSequenceTokens && seqLenKv > maxShortSequenceTokens &&
           (headDim >= minHeadDimWithoutTailLimit || seqLenQ % queryTileRows <= queryTileRows / tailTileDivisor);
}

QFA_WORKLOAD_INLINE uint64_t QfaCausalTaskCost(uint64_t tasks, uint32_t queryTiles, uint32_t tilesPerKv)
{
    const uint64_t complete = tasks / queryTiles;
    const uint64_t tail = tasks % queryTiles;
    const uint64_t fullGroups = queryTiles / tilesPerKv, fullTail = queryTiles % tilesPerKv;
    const uint64_t tailGroups = tail / tilesPerKv, tailTail = tail % tilesPerKv;
    const uint64_t headCost = tilesPerKv * fullGroups * (fullGroups + 1U) / 2U + fullTail * (fullGroups + 1U);
    const uint64_t tailCost = tilesPerKv * tailGroups * (tailGroups + 1U) / 2U + tailTail * (tailGroups + 1U);
    return complete * headCost + tailCost + QFA_CORE_WORK_SETUP_STAGES * tasks;
}

QFA_WORKLOAD_INLINE uint64_t QfaCausalTaskBoundary(uint32_t core, const QuantFlashAttnTilingDataInt8& t, bool bnsd,
                                                   bool sg)
{
    const uint32_t queryTiles = bnsd || !sg ? t.seqLenQ / t.blockM : t.totalTilesM;
    const uint32_t tilesPerKv = (bnsd || !sg ? 1U : t.groupSize) * t.blockN / t.blockM;
    const uint64_t target = QfaCausalTaskCost(t.totalTasks, queryTiles, tilesPerKv) * core / t.usedCoreNum;
    uint64_t lo = 0U, hi = t.totalTasks;
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2U;
        if (QfaCausalTaskCost(mid, queryTiles, tilesPerKv) < target)
            lo = mid + 1U;
        else
            hi = mid;
    }
    return lo;
}
struct QfaCoreTaskRange {
    uint64_t begin, end;
    uint32_t step;
};
QFA_WORKLOAD_INLINE QfaCoreTaskRange QfaDenseTaskRange(const QuantFlashAttnTilingDataInt8& t, uint32_t core)
{
    const bool bnsd = t.layoutQ == QFA_LAYOUT_BNSD && t.layoutKv == QFA_LAYOUT_BNSD && t.layoutOut == QFA_LAYOUT_BNSD;
    const bool sg = t.layoutQ != QFA_LAYOUT_BNSD;
    // The D72 layout keeps each head local even when the normal SG test fails.
    const bool contiguous = t.headDim == QFA_HEAD_DIM_72 || sg;
    QfaCoreTaskRange r{contiguous ? static_cast<uint64_t>(t.totalTasks) * core / t.usedCoreNum : core,
                       contiguous ? static_cast<uint64_t>(t.totalTasks) * (core + 1U) / t.usedCoreNum : t.totalTasks,
                       contiguous ? 1U : t.usedCoreNum};
    const uint32_t group = bnsd || !sg ? 1U : t.groupSize;
    if (t.layoutQ <= QFA_LAYOUT_BSND && t.layoutKv <= QFA_LAYOUT_BSND && t.maskMode != 0U && t.seqLenQ == t.seqLenKv &&
        static_cast<uint64_t>(t.seqLenQ) * group % t.blockM == 0U &&
        static_cast<uint64_t>(t.blockN) * group % t.blockM == 0U) {
        r = {QfaCausalTaskBoundary(core, t, bnsd, sg), QfaCausalTaskBoundary(core + 1U, t, bnsd, sg), 1U};
    }
    return r;
}
// Keep the per-core workspace reservation and the kernel cache policy in sync.
template <typename SequenceLength>
QFA_WORKLOAD_INLINE bool QfaUseKScaleCache(SequenceLength seqLenQ, SequenceLength seqLenKv, uint32_t headDim,
                                           uint64_t groupedQueryRows, uint32_t queryTileRows, uint32_t kvTileTokens)
{
    constexpr uint32_t maxShortQueryHeadDim = 32U;
    constexpr uint32_t minRepeatedQueryLength = 128U;
    constexpr uint32_t longQueryThreshold = 512U;
    // Only M512 at small D has enough L1 space to retain all the scales on this short-input path.
    const bool shortRepeatedQueries = queryTileRows == QFA_LARGE_QUERY_TILE_ROWS && headDim <= maxShortQueryHeadDim &&
                                      seqLenQ > minRepeatedQueryLength && seqLenKv > minRepeatedQueryLength &&
                                      groupedQueryRows >= 2U * queryTileRows; // Reuse K scales across Q tiles.
    const bool singlePanelReuse = seqLenKv <= kvTileTokens && groupedQueryRows > queryTileRows;
    return (seqLenQ > longQueryThreshold || shortRepeatedQueries || singlePanelReuse) &&
           seqLenKv <= QFA_K_SCALE_CACHE_TOKENS;
}

// The existing validity bitmap caches 32 immutable KV-scale panels. A resident-Q pipeline streams later panels
// through its original GM slots.
QFA_WORKLOAD_INLINE uint32_t QfaKScaleWorkspaceElements(bool cached, bool prefix, uint32_t slots, uint32_t n)
{
    return prefix ? (QFA_K_SCALE_PREFIX_PANELS + slots) * n : (cached ? QFA_K_SCALE_CACHE_TOKENS : slots * n);
}
QFA_WORKLOAD_INLINE bool QfaKScaleCachedTile(bool cached, bool prefix, uint32_t start, uint32_t n)
{
    return cached && (!prefix || start < QFA_K_SCALE_PREFIX_PANELS * n);
}
QFA_WORKLOAD_INLINE uint32_t QfaKScaleGmOffset(bool cached, bool prefix, uint32_t start, uint32_t slot, uint32_t n)
{
    return QfaKScaleCachedTile(cached, prefix, start, n) ? start :
                                                           (prefix ? QFA_K_SCALE_PREFIX_PANELS * n : 0U) + slot * n;
}
// Sum floor((a*i+b)/m), i in [0,n). Euclidean reduction avoids an O(S) loop.
QFA_WORKLOAD_INLINE uint64_t QfaFloorSum(uint64_t n, uint64_t m, uint64_t a, uint64_t b)
{
    uint64_t sum = 0U;
    while (true) {
        if (a >= m) {
            sum += (n * (n - 1U) / 2U) * (a / m);
            a %= m;
        }
        if (b >= m) {
            sum += n * (b / m);
            b %= m;
        }
        const uint64_t y = a * n + b;
        if (y < m)
            return sum;
        n = y / m;
        b = y % m;
        const uint64_t oldM = m;
        m = a;
        a = oldM;
    }
}
QFA_WORKLOAD_INLINE uint64_t QfaHeadWork(uint64_t count, uint64_t tiles, uint32_t m, int64_t delta,
                                         uint64_t denominator, uint64_t kvTiles)
{
    if (count == 0U)
        return 0U;
    const uint64_t first = delta < 0 ? static_cast<uint64_t>(-delta) / m + 1U : 1U;
    if (first > count)
        return QFA_CORE_WORK_SETUP_STAGES * count;
    const uint64_t base = static_cast<uint64_t>(static_cast<int64_t>(first * m) + delta);
    uint64_t work = QfaFloorSum(count - first + 1U, denominator, m, base + denominator - 1U);
    if (count == tiles) {
        const uint64_t last = static_cast<uint64_t>(static_cast<int64_t>(count * m) + delta);
        const uint64_t paddedTiles = (last + denominator - 1U) / denominator;
        if (paddedTiles > kvTiles)
            work -= paddedTiles - kvTiles;
    }
    return work + QFA_CORE_WORK_SETUP_STAGES * count;
}

// Exact number of visible KV tiles, plus two pipeline stages per Q task. Q tasks traverse S1G; the final task may
// contain a partial M tile.
QFA_WORKLOAD_INLINE uint64_t QfaSequenceWork(uint64_t tasks, uint32_t sq, uint32_t sk, uint32_t group, uint32_t m,
                                             uint32_t n, bool causal)
{
    if (tasks == 0U || sq == 0U)
        return 0U;
    const uint64_t kvTiles = (static_cast<uint64_t>(sk) + n - 1U) / n;
    if (!causal)
        return tasks * (kvTiles + QFA_CORE_WORK_SETUP_STAGES);
    const uint64_t tiles = (static_cast<uint64_t>(sq) * group + m - 1U) / m;
    const int64_t delta = static_cast<int64_t>(group) * (static_cast<int64_t>(sk) - sq);
    const uint64_t denominator = static_cast<uint64_t>(group) * n;
    return (tasks / tiles) * QfaHeadWork(tiles, tiles, m, delta, denominator, kvTiles) +
           QfaHeadWork(tasks % tiles, tiles, m, delta, denominator, kvTiles);
}

// Packed sequence lengths and the corresponding query tile counts.
struct QfaSequenceDescriptor {
    uint32_t qStart = 0U, kStart = 0U, sq = 0U, sk = 0U, tiles = 0U, tasks = 0U;
};
struct QfaScheduleParameters {
    uint32_t batchSize, usedCoreNum, groupSize, blockM, blockN, maskMode;
};
template <typename Reader>
class QfaTaskRangePlanner {
    using Sequence = QfaSequenceDescriptor;

public:
    QFA_WORKLOAD_INLINE QfaTaskRangePlanner(const QfaScheduleParameters& params, Reader& reader)
        : t_(params),
          reader_(reader)
    {}
    QFA_WORKLOAD_INLINE void Init(uint32_t core)
    {
        totalTasks_ = 0U;
        uint64_t totalWork = 0U;
        for (uint32_t b = 0; b < t_.batchSize; ++b) {
            const Sequence s = Read(b);
            totalTasks_ += s.tasks;
            totalWork += Work(s, s.tasks);
        }
        begin_ = core == 0U ? 0U : Boundary(totalWork * core / t_.usedCoreNum);
        end_ = core + 1U == t_.usedCoreNum ? totalTasks_ : Boundary(totalWork * (core + 1U) / t_.usedCoreNum);
    }
    QFA_WORKLOAD_INLINE uint64_t Begin() const
    {
        return begin_;
    }
    QFA_WORKLOAD_INLINE uint64_t End() const
    {
        return end_;
    }

private:
    QFA_WORKLOAD_INLINE Sequence Read(uint32_t batch)
    {
        return reader_.ReadSequence(batch);
    }
    QFA_WORKLOAD_INLINE uint64_t Work(const Sequence& s, uint64_t tasks) const
    {
        return QfaSequenceWork(tasks, s.sq, s.sk, t_.groupSize, t_.blockM, t_.blockN, t_.maskMode != 0U);
    }
    QFA_WORKLOAD_INLINE uint64_t Boundary(uint64_t target)
    {
        uint64_t taskBase = 0U, workBase = 0U;
        for (uint32_t b = 0; b < t_.batchSize; ++b) {
            const Sequence s = Read(b);
            const uint64_t cost = Work(s, s.tasks);
            if (cost > 0U && target <= workBase + cost) {
                const uint64_t localTarget = target - workBase;
                uint32_t lo = 0U, hi = s.tasks;
                while (lo < hi) {
                    const uint32_t mid = lo + (hi - lo) / 2U;
                    if (Work(s, mid) < localTarget)
                        lo = mid + 1U;
                    else
                        hi = mid;
                }
                if (lo > QFA_LAYOUT_BNSD && localTarget - Work(s, lo - 1U) < Work(s, lo) - localTarget)
                    --lo;
                return taskBase + lo;
            }
            taskBase += s.tasks;
            workBase += cost;
        }
        return taskBase;
    }
    QfaScheduleParameters t_;
    Reader& reader_;
    uint64_t begin_ = 0U, end_ = 0U, totalTasks_ = 0U;
};
} // namespace BaseApi::Int8GqaTnd
#ifdef QFA_WORKLOAD_LOCAL_INLINE
#undef QFA_WORKLOAD_INLINE
#undef QFA_WORKLOAD_LOCAL_INLINE
#endif
namespace BaseApi {
using BaseApi::Int8GqaTnd::QfaFloorSum;
using BaseApi::Int8GqaTnd::QfaHeadWork;
using BaseApi::Int8GqaTnd::QfaSequenceWork;
} // namespace BaseApi
#endif
