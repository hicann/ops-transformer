/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_BAND_GROUPED_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_BAND_GROUPED_H_

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <vector>
#include "../candidate_types.h"
#include "../fag_deter_utils.h"
#include "candidates.h"

namespace optiling {
namespace fag_pto_arch35 {
using BandCoreLoads = std::array<int64_t, MAX_CORE_NUM>;

// Bound host selection work, not sequence dimensions or device geometry. One candidate
// visits H*N column owners; evaluating every H costs N*K*(K+1)/2 visits at most.
constexpr int64_t MHA_BAND_MAX_COLUMN_EVALUATIONS = 1'000'000;

struct BandPhysicalWindow {
    int64_t s1;
    int64_t s2;
    int64_t pre;
    int64_t next;
    int64_t rowOffset;
    int64_t colOffset;
};

struct BandCacheFootprint {
    uint64_t perHeadBytes = 0;
    uint64_t sharedBytes = 0;
    uint64_t l2Bytes = 0;
};

inline uint64_t SaturatingBandAdd(uint64_t lhs, uint64_t rhs)
{
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    return lhs > limit - rhs ? limit : lhs + rhs;
}

inline uint64_t SaturatingBandMultiply(uint64_t lhs, uint64_t rhs)
{
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    return rhs != 0 && lhs > limit / rhs ? limit : lhs * rhs;
}

inline BandCacheFootprint CalcBandCacheFootprint(int64_t k, int64_t s1, int64_t s2, int64_t d, int64_t dv,
                                                 uint64_t inputBytes, uint64_t outputBytes, bool enablePreSfmg,
                                                 uint64_t maskBytes, uint64_t l2Bytes)
{
    BandCacheFootprint result;
    if (k <= 0 || s1 <= 0 || s2 <= 0 || d <= 0 || dv <= 0 || l2Bytes == 0 || inputBytes != 2 ||
        (outputBytes != 2 && outputBytes != 4)) {
        return result;
    }
    const uint64_t align = 16;
    const uint64_t dAligned = SaturatingBandMultiply(SaturatingBandAdd(d, align - 1) / align, align);
    const uint64_t dvAligned = SaturatingBandMultiply(SaturatingBandAdd(dv, align - 1) / align, align);
    const uint64_t readRowBytes = SaturatingBandMultiply(SaturatingBandAdd(d, dv), inputBytes);
    // Retain the upstream 64-byte statistics budget as a conservative capacity proxy.
    // PTO loads one FP32 LSE per row; it does not precompute SFMG in GM.
    const uint64_t statisticsBytes = NUM_TWO * 32;
    const uint64_t sfmgBytes = enablePreSfmg ? 4 : SaturatingBandMultiply(dv, outputBytes);
    const uint64_t queryRowBytes =
        SaturatingBandAdd(SaturatingBandAdd(readRowBytes, SaturatingBandMultiply(dAligned, 4)),
                          SaturatingBandAdd(statisticsBytes, sfmgBytes));
    const uint64_t kvRowBytes =
        SaturatingBandAdd(readRowBytes, SaturatingBandMultiply(SaturatingBandAdd(dAligned, dvAligned), 4));
    result.perHeadBytes =
        SaturatingBandAdd(SaturatingBandMultiply(s1, queryRowBytes), SaturatingBandMultiply(s2, kvRowBytes));
    // At a group transition each core can retain one retiring task and prefetch Q/dY.
    // This is logical GM payload, not a claim that these bytes all reside in L2.
    const uint64_t tileRows = static_cast<uint64_t>(ConstAxisTemplateNum::NUM128);
    const uint64_t transitionBytes = SaturatingBandMultiply(
        k, SaturatingBandAdd(
               SaturatingBandMultiply(std::min<uint64_t>(s1, tileRows), SaturatingBandAdd(queryRowBytes, readRowBytes)),
               SaturatingBandMultiply(std::min<uint64_t>(s2, tileRows), kvRowBytes)));
    result.sharedBytes = SaturatingBandAdd(maskBytes, transitionBytes);
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    if (dAligned == limit || dvAligned == limit || result.perHeadBytes == limit || result.sharedBytes == limit) {
        return {};
    }
    result.l2Bytes = l2Bytes;
    return result;
}

inline uint64_t CalcBandSpillBytes(int64_t b, int64_t heads, const BandCacheFootprint& cache)
{
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    const uint64_t fullBytes = SaturatingBandAdd(SaturatingBandMultiply(heads, cache.perHeadBytes), cache.sharedBytes);
    const int64_t tailHeads = b % heads;
    const uint64_t tailBytes =
        tailHeads == 0 ? 0 :
                         SaturatingBandAdd(SaturatingBandMultiply(tailHeads, cache.perHeadBytes), cache.sharedBytes);
    if (fullBytes == limit || tailBytes == limit) {
        return limit;
    }
    const uint64_t fullSpill = fullBytes > cache.l2Bytes ? fullBytes - cache.l2Bytes : 0;
    const uint64_t tailSpill = tailBytes > cache.l2Bytes ? tailBytes - cache.l2Bytes : 0;
    return SaturatingBandAdd(SaturatingBandMultiply(b / heads, fullSpill), tailSpill);
}

struct BandRoundActivity {
    int64_t rounds = 0;
    int64_t activeRounds = 0;
    int64_t activeRuns = 0;
    bool firstActive = false;
    bool lastActive = false;
};

inline BandRoundActivity CalcBandGroupActivity(int64_t k, int64_t m, int64_t heads,
                                               const std::vector<int64_t>& firstRows,
                                               const std::vector<int64_t>& heights)
{
    BandRoundActivity result;
    if (heads == 0) {
        return result;
    }
    const int64_t n = heights.size();
    const int64_t activeCores = std::min(k, heads * m);
    const int64_t waves = PositiveCeilDiv(heads * n, activeCores);
    result.rounds = waves * m;
    std::vector<std::pair<int64_t, int64_t>> intervals;
    intervals.reserve(NUM_TWO * k);
    for (int64_t wave = 0; wave < waves; ++wave) {
        intervals.clear();
        const int64_t packedBegin = wave * activeCores;
        const int64_t packedEnd = std::min(packedBegin + activeCores, heads * n);
        // Repeated heads at one column contribute the same circular row interval.
        for (int64_t col = packedBegin / heads; col <= (packedEnd - 1) / heads; ++col) {
            if (heights[col] == 0) {
                continue;
            }
            if (heights[col] == m) {
                intervals = {{0, m}};
                break;
            }
            int64_t start = (firstRows[col] - col) % m;
            if (start < 0) {
                start += m;
            }
            const int64_t end = start + heights[col];
            intervals.emplace_back(start, std::min(end, m));
            if (end > m) {
                intervals.emplace_back(0, end - m);
            }
        }
        std::sort(intervals.begin(), intervals.end());
        int64_t unionLength = 0;
        int64_t unionRuns = 0;
        int64_t end = -1;
        for (const auto& interval : intervals) {
            if (interval.first > end) {
                unionLength += interval.second - interval.first;
                ++unionRuns;
                end = interval.second;
            } else if (interval.second > end) {
                unionLength += interval.second - end;
                end = interval.second;
            }
        }
        const bool first = !intervals.empty() && intervals.front().first == 0;
        result.activeRounds += unionLength;
        result.activeRuns += unionRuns - static_cast<int64_t>(result.lastActive && first);
        if (wave == 0) {
            result.firstActive = first;
        }
        result.lastActive = end == m;
    }
    return result;
}

inline BandRoundActivity AccumulateBandActivity(const BandRoundActivity& full, int64_t fullCount,
                                                const BandRoundActivity& tail)
{
    BandRoundActivity result;
    result.rounds = fullCount * full.rounds + tail.rounds;
    result.activeRounds = fullCount * full.activeRounds + tail.activeRounds;
    result.activeRuns = fullCount * full.activeRuns + tail.activeRuns;
    if (fullCount > 1 && full.firstActive && full.lastActive) {
        result.activeRuns -= fullCount - 1;
    }
    if (fullCount > 0 && tail.rounds > 0 && full.lastActive && tail.firstActive) {
        --result.activeRuns;
    }
    result.firstActive = fullCount > 0 ? full.firstActive : tail.firstActive;
    result.lastActive = tail.rounds > 0 ? tail.lastActive : full.lastActive;
    return result;
}

inline int64_t FloorDivideSigned(int64_t value, int64_t divisor)
{
    return value / divisor - static_cast<int64_t>(value % divisor < 0);
}

inline std::vector<int64_t> CalcBandPhysicalColumnHeights(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q,
                                                          int64_t b, const BandPhysicalWindow& window,
                                                          std::vector<int64_t>* firstRows = nullptr)
{
    if (k <= 0 || k > MAX_CORE_NUM || b <= 1 || m <= 0 || n <= 0 || p <= 0 || q <= 0 || window.s1 <= 0 ||
        window.s2 <= 0 || window.pre < INT32_MIN || window.pre > INT32_MAX || window.next < INT32_MIN ||
        window.next > INT32_MAX || window.rowOffset < 0 || window.colOffset < 0) {
        return {};
    }
    const int64_t maxHeads = std::min(k, b - 1);
    const int64_t ownerVisits = maxHeads * (maxHeads + 1) / NUM_TWO;
    if (n > MHA_BAND_MAX_COLUMN_EVALUATIONS / ownerVisits) {
        return {};
    }
    constexpr int64_t cubeBase = static_cast<int64_t>(ConstAxisTemplateNum::NUM128);
    constexpr int64_t sparseAlign = 64;
    const int64_t physicalRows = PositiveCeilDiv(window.s1, cubeBase);
    const int64_t physicalCols = PositiveCeilDiv(window.s2, cubeBase);
    if (physicalRows >= std::numeric_limits<uint32_t>::max() || physicalCols >= std::numeric_limits<uint32_t>::max() ||
        window.rowOffset > physicalRows || window.colOffset > physicalCols || m > physicalRows - window.rowOffset ||
        n > physicalCols - window.colOffset) {
        return {};
    }
    std::vector<int64_t> heights(n, 0);
    if (firstRows != nullptr) {
        firstRows->assign(n, 0);
    }
    p = std::min(p, m);
    q = std::min(q, n);
    for (int64_t col = 0; col < n; ++col) {
        const int64_t colLeft = (col + window.colOffset) * cubeBase;
        const int64_t colRight = std::min(colLeft + cubeBase, window.s2);
        if (colLeft >= window.s2 || window.s1 + window.next <= colLeft) {
            continue;
        }
        const int64_t alignedRight = PositiveCeilDiv(colRight, sparseAlign) * sparseAlign;
        // Exact IsValidForDeter BAND inequalities, including real Q/KV tails:
        // rowEnd + next > colLeft; rowStart - pre < alignUp(colRight, 64).
        // colLeft is 128-aligned, so its comparison cancels the other AlignTo64.
        const int64_t firstRow = std::max(window.rowOffset, FloorDivideSigned(colLeft - window.next, cubeBase));
        const int64_t lastRow = std::min({window.rowOffset + m - 1, PositiveCeilDiv(window.s1, cubeBase) - 1,
                                          FloorDivideSigned(window.pre + alignedRight - 1, cubeBase)});
        if (lastRow < firstRow) {
            continue;
        }
        // The original scalar mapping must cover every actual task counted here.
        // Otherwise neither its cost nor a change in coverage is an optimization baseline.
        const int64_t firstGeometricRow = std::max<int64_t>(0, col - q + 1) + window.rowOffset;
        const int64_t endGeometricRow = std::min(m, col + p) + window.rowOffset;
        if (firstRow < firstGeometricRow || lastRow >= endGeometricRow) {
            return {};
        }
        heights[col] = lastRow - firstRow + 1;
        if (firstRows != nullptr) {
            (*firstRows)[col] = firstRow - window.rowOffset;
        }
    }
    return heights;
}

inline BandCoreLoads CalcBandGroupCoreLoads(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t heads,
                                            const std::vector<int64_t>& columnHeights = {})
{
    BandCoreLoads loads{};
    if (heads == 0) {
        return loads;
    }
    const int64_t activeCores = std::min(k, heads * m);
    for (int64_t column = 0; column < n; ++column) {
        const int64_t firstRow = std::max<int64_t>(0, column - q + 1);
        const int64_t endRow = std::min(m, column + p);
        const int64_t height = columnHeights.empty() ? std::max<int64_t>(0, endRow - firstRow) : columnHeights[column];
        for (int64_t head = 0; head < heads; ++head) {
            loads[(column * heads + head) % activeCores] += height;
        }
    }
    return loads;
}

inline BandCoreLoads AccumulateRotatedBandGroups(const BandCoreLoads& groupLoads, int64_t k, int64_t heads,
                                                 int64_t groupCount)
{
    BandCoreLoads loads{};
    const int64_t orbitCount = PositiveGcd(k, heads);
    const int64_t period = k / orbitCount;
    const int64_t fullPeriods = groupCount / period;
    const int64_t remainder = groupCount % period;
    // Each orbit is a circular window of group rotations. Its length is at most K,
    // even for very large B*N2; selection never enumerates batches or tile rows.
    for (int64_t root = 0; root < orbitCount; ++root) {
        int64_t orbitSum = 0;
        int64_t window = 0;
        for (int64_t i = 0; i < period; ++i) {
            const int64_t core = (root + i * heads) % k;
            orbitSum += groupLoads[core];
            if (i < remainder) {
                window += groupLoads[core];
            }
        }
        for (int64_t i = 0; i < period; ++i) {
            const int64_t core = (root + i * heads) % k;
            loads[core] = fullPeriods * orbitSum + window;
            if (remainder != 0) {
                window -= groupLoads[core];
                window += groupLoads[(root + (i + remainder) * heads) % k];
            }
        }
    }
    return loads;
}

inline DeterBandScheduleResult SelectGroupedDeterBandSchedule(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q,
                                                              int64_t b, int64_t hostMaxRound,
                                                              const std::vector<int64_t>& columnHeights = {},
                                                              const std::vector<int64_t>& firstRows = {},
                                                              const BandCacheFootprint& cache = {})
{
    DeterBandScheduleResult result;
    if (k <= 0 || k > MAX_CORE_NUM || m <= 0 || n <= 0 || b <= 1 || p <= 0 || q <= 0 ||
        m >= std::numeric_limits<uint32_t>::max() || n >= std::numeric_limits<uint32_t>::max()) {
        return result;
    }
    if (columnHeights.size() != static_cast<size_t>(n) || firstRows.size() != columnHeights.size() ||
        cache.l2Bytes == 0 || cache.perHeadBytes == 0) {
        return result;
    }
    for (int64_t col = 0; col < n; ++col) {
        if (firstRows[col] < 0 || columnHeights[col] < 0 || firstRows[col] > m ||
            columnHeights[col] > m - firstRows[col]) {
            return result;
        }
    }
    p = std::min(p, m);
    q = std::min(q, n);
    // For wide bands, the legacy complete K-head chunks have equal valid work,
    // and their remainder uses CalDenseIndex. Narrow-band tails use a different
    // mapping; retain that path until its complete cost is modeled.
    if (p + q <= m || m > std::numeric_limits<int64_t>::max() / n ||
        b > std::numeric_limits<int64_t>::max() / (m * n)) {
        return result;
    }
    const int64_t legacyRound = CalcLegacyDeterBandMaxRound(k, m, n, p, q, b, hostMaxRound);
    if (legacyRound <= 0 || legacyRound >= std::numeric_limits<uint32_t>::max()) {
        return result;
    }
    const int64_t maxHeads = std::min(k, b - 1);
    const int64_t searchOwners = maxHeads * (maxHeads + 1) / NUM_TWO;
    if (n > MHA_BAND_MAX_COLUMN_EVALUATIONS / searchOwners) {
        return result;
    }

    const BandCoreLoads single = CalcBandGroupCoreLoads(k, m, n, p, q, 1, columnHeights);
    int64_t validBlocks = 0;
    for (int64_t core = 0; core < k; ++core) {
        validBlocks += single[core];
    }
    const BandCoreLoads legacyTail = CalcBandGroupCoreLoads(k, m, n, p, q, b % k, columnHeights);
    int64_t legacyMaxLoad = 0;
    for (int64_t core = 0; core < k; ++core) {
        legacyMaxLoad = std::max(legacyMaxLoad, (b / k) * validBlocks + legacyTail[core]);
    }

    // Every candidate uses M-row waves and at most min(K, B*M) owners.
    // Per-group ceilings cannot beat this whole-operator wave lower bound.
    const int64_t roundLowerBound = PositiveCeilDiv(n * b, std::min(k, b * m)) * m;
    const int64_t loadLowerBound = PositiveCeilDiv(validBlocks * b, k);
    int64_t bestMaxLoad = std::numeric_limits<int64_t>::max();
    // Lexicographic cost: rounds, busiest core, spill, nonempty rounds, pipeline drain, H.
    std::array<uint64_t, 6> bestCost;
    bestCost.fill(std::numeric_limits<uint64_t>::max());
    for (int64_t heads = 1; heads <= maxHeads; ++heads) {
        const int64_t fullCount = b / heads;
        const int64_t tailHeads = b % heads;
        const int64_t activeCores = std::min(k, heads * m);
        const int64_t fullRound = PositiveCeilDiv(n * heads, activeCores) * m;
        const int64_t tailRound = tailHeads == 0 ? 0 : PositiveCeilDiv(n * tailHeads, std::min(k, tailHeads * m)) * m;
        // The complete operator bound includes the independently drained tail group.
        // Requiring a shorter group span gives a real locality improvement, without
        // a shape whitelist or a fixed-D/cache-capacity threshold.
        if (tailRound > legacyRound || fullRound > (legacyRound - tailRound) / fullCount || fullRound >= validBlocks) {
            continue;
        }
        const int64_t totalRound = fullCount * fullRound + tailRound;
        const uint64_t spillBytes = CalcBandSpillBytes(b, heads, cache);
        if (spillBytes == std::numeric_limits<uint64_t>::max() ||
            (result.mode != DeterBandScheduleMode::DISABLED &&
             (totalRound > result.maxRound ||
              (totalRound == result.maxRound && bestMaxLoad == loadLowerBound && spillBytes > bestCost[2])))) {
            continue;
        }
        const BandCoreLoads fullLoads = CalcBandGroupCoreLoads(k, m, n, p, q, heads, columnHeights);
        BandCoreLoads loads = AccumulateRotatedBandGroups(fullLoads, k, heads, fullCount);
        const BandCoreLoads tailLoads = CalcBandGroupCoreLoads(k, m, n, p, q, tailHeads, columnHeights);
        int64_t maxLoad = 0;
        for (int64_t core = 0; core < k; ++core) {
            loads[core] += tailLoads[(core + fullCount * heads) % k];
            maxLoad = std::max(maxLoad, loads[core]);
        }
        if (maxLoad > legacyMaxLoad) {
            continue;
        }
        const auto activity =
            AccumulateBandActivity(CalcBandGroupActivity(k, m, heads, firstRows, columnHeights), fullCount,
                                   CalcBandGroupActivity(k, m, tailHeads, firstRows, columnHeights));
        // Invalid rounds still synchronize; count rounds carrying real work and the one-round drain.
        const std::array<uint64_t, 6> cost{static_cast<uint64_t>(totalRound),
                                           static_cast<uint64_t>(maxLoad),
                                           spillBytes,
                                           static_cast<uint64_t>(activity.activeRounds),
                                           static_cast<uint64_t>(activity.activeRounds + activity.activeRuns),
                                           static_cast<uint64_t>(heads)};
        if (cost >= bestCost) {
            continue;
        }
        bestCost = cost;
        result.mode = DeterBandScheduleMode::GROUPED;
        result.maxRound = totalRound;
        result.m = m;
        result.n = n;
        result.p = p;
        result.q = q;
        bestMaxLoad = maxLoad;
        auto& plan = result.groupedSchedule;
        plan = {};
        plan.phaseCount = 1;
        plan.phases[0].m = m;
        plan.phases[0].n = n;
        plan.k = k;
        plan.b = b;
        plan.g = 1;
        plan.groupsPerChunk = heads;
        plan.fullRounds = fullRound;
        plan.tailRounds = tailRound;
        plan.totalRounds = totalRound;
        // Two bounds alone cannot stop the search: a later H may reduce active rounds.
        // Stop only when every cost has reached its absolute lower bound.
        if (totalRound == roundLowerBound && maxLoad == loadLowerBound && spillBytes == 0 &&
            activity.activeRounds == loadLowerBound && activity.activeRuns == 1) {
            break;
        }
    }
    return result;
}

} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_BAND_GROUPED_H_
