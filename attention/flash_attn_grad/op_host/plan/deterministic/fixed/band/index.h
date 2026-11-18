/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_BAND_INDEX_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_BAND_INDEX_H_

// BAND round and index calculations.
#include "schedule_types.h"
#include "../fag_deter_utils.h"

namespace optiling {
namespace fag_pto_arch35 {
// Round/index arithmetic used by candidate evaluation. Selection stays in selector.h.

// Core-budget helpers do not receive G; use the minimum GQA head count
// when checking the geometry-only part of overlap eligibility.
constexpr int64_t GQA_BAND_MIN_QUERY_HEADS_PER_KV = 2;

// Independent KV groups share the core budget; each needs floor(m/2)
// triangle cores and at least one rectangle lane. The tail uses its actual group count.
inline bool CanUseGqaBandOverlap(int64_t k, int64_t groups, int64_t g, int64_t m, int64_t n)
{
    return k > 0 && groups > 0 && g > 1 && m >= 2 && n > 0 && groups <= k / (m / 2 + 1);
}

// The side-(m-1) triangle fits in floor(m/2) cores. For odd m its
// paired columns need m rows; for even m they finish one row before the stripe.
inline int64_t GqaBandOverlapTriangleRounds(int64_t m, int64_t g)
{
    return m >= 2 && g > 1 ? (m - 1 + m % 2) * g : 0;
}

inline int64_t GqaBandOverlapCoreBudget(int64_t k, int64_t groups, int64_t m, int64_t n, int64_t group)
{
    if (!CanUseGqaBandOverlap(k, groups, GQA_BAND_MIN_QUERY_HEADS_PER_KV, m, n) || group < 0 || group >= groups) {
        return 0;
    }
    const int64_t available = k / groups + (group < k % groups ? 1 : 0);
    const int64_t useful = m / 2 + (m < n ? m : n);
    return available < useful ? available : useful;
}

// group is zero-based; localCore is one-based. Capped, unused cores stay empty.
inline bool GqaBandOverlapCoreIndex(int64_t k, int64_t groups, int64_t m, int64_t n, int64_t core, int64_t& group,
                                    int64_t& localCore, int64_t& localK)
{
    group = -1;
    localCore = 0;
    localK = 0;
    if (!CanUseGqaBandOverlap(k, groups, GQA_BAND_MIN_QUERY_HEADS_PER_KV, m, n) || core < 1 || core > k) {
        return false;
    }
    const int64_t extra = k % groups;
    const int64_t useful = m / 2 + (m < n ? m : n);
    const int64_t base = k / groups;
    const int64_t highBudget = base + 1 < useful ? base + 1 : useful;
    const int64_t lowBudget = base < useful ? base : useful;
    const int64_t highCores = extra * highBudget;
    if (core > highCores + (groups - extra) * lowBudget) {
        return false;
    }
    if (core <= highCores) {
        group = (core - 1) / highBudget;
        localCore = (core - 1) % highBudget + 1;
        localK = highBudget;
    } else {
        group = extra + (core - highCores - 1) / lowBudget;
        localCore = (core - highCores - 1) % lowBudget + 1;
        localK = lowBudget;
    }
    return true;
}

// A physical diagonal column stays on one core for all G query heads.
// Distinct columns imply distinct dQ rows as well as distinct shared dK/dV addresses.
inline void GqaBandDiagonalColumnIndex(int64_t k, int64_t size, int64_t groups, int64_t g, int64_t core, int64_t round,
                                       int64_t& w, int64_t& x, int64_t& y)
{
    w = -1;
    if (k <= 0 || size <= 0 || groups <= 0 || g <= 1 || core < 1 || core > k || round < 1) {
        return;
    }
    const int64_t column = (round - 1) / g * k + core;
    if (column > groups * size) {
        return;
    }
    w = (column - 1) / size * g + (round - 1) % g + 1;
    x = (column - 1) % size + 1;
    y = x;
}

// Decode folded phase tags without treating whole-band p/q as translations.
inline GqaBandDomain DecodeGqaBandFoldedDomain(const GqaBandPhase& phase)
{
    const bool upper = phase.kind == GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER;
    const bool lower = phase.kind == GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER;
    const bool wholeBand = phase.kind == GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER;
    return {phase.m,
            phase.n,
            wholeBand ? phase.rowOffset : (upper ? 1 : phase.colOffset),
            wholeBand ? phase.colOffset : (upper ? phase.rowOffset : 1),
            lower ? phase.rowOffset : 0,
            upper ? phase.colOffset : 0};
}

inline bool CanUseGqaBandFoldedColumnOwner(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p, int64_t q)
{
    return k > 0 && b > 0 && g > 1 && m > 0 && n > 0 && p > 0 && p <= m && q > 0 && q <= n;
}

// A virtual column v owns every physical y with (y-1)%m == v-1. All cores
// use the same head and diagonal offset t=y-x in a round. Distinct virtual
// columns in one group therefore have distinct rows and distinct physical
// columns, even when active columns outnumber G. Each physical column's t
// interval is contiguous; head-inner order retains one uninterrupted KV run.
inline void GqaBandFoldedColumnOwnerIndex(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p, int64_t q,
                                          int64_t core, int64_t round, int64_t& w, int64_t& x, int64_t& y)
{
    w = -1;
    if (!CanUseGqaBandFoldedColumnOwner(k, b, g, m, n, p, q) || core < 1 || core > k || round < 1) {
        return;
    }
    // Products are bounded by the common host arithmetic guard before serialization.
    const int64_t size = m < n ? m : n;
    const int64_t width = p + q - 1;
    const int64_t roundsPerBatch = width * g;
    const int64_t batch = (round - 1) / roundsPerBatch;
    const int64_t columns = b * size;
    const int64_t batches = columns / k + (columns % k != 0 ? 1 : 0);
    if (batch >= batches) {
        return;
    }
    const int64_t column = batch * k + core - 1;
    if (column >= columns) {
        return;
    }
    const int64_t virtualColumn = column % size + 1;
    const int64_t diagonal = (round - 1) / g % width + 1 - p;
    const int64_t rowRemainder = (virtualColumn - diagonal - 1) % m;
    x = rowRemainder + (rowRemainder < 0 ? m : 0) + 1;
    y = x + diagonal;
    if (y < 1 || y > n) {
        return;
    }
    w = (column / size) * g + (round - 1) % g + 1;
}

// Each KV group gets min(available cores per group, query heads) lanes.
// A lane owns mirrored columns in each 2*lanes block. This balances rising
// and falling band edges without depending on sequence lengths or head powers.
inline int64_t GqaBandColumnOwnerLanes(int64_t k, int64_t b, int64_t g)
{
    if (k <= 0 || b <= 0 || b > k || g <= 1) {
        return 0;
    }
    return k / b < g ? k / b : g;
}

inline bool CanUseGqaBandColumnOwner(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p, int64_t q)
{
    return GqaBandColumnOwnerLanes(k, b, g) > 0 && m > 0 && n > 0 && p > 0 && p <= m && q > 0 && q <= n;
}

// Sum clamp(first + i*step, 0, limit), 0 <= i < count. The host's common
// arithmetic guard bounds geometry products before this device helper is used.
inline int64_t GqaBandClippedProgressionSum(int64_t first, int64_t step, int64_t count, int64_t limit)
{
    if (count <= 0) {
        return 0;
    }
    const int64_t last = first + (count - 1) * step;
    if (last <= 0) {
        return 0;
    }
    if (first >= limit) {
        return count * limit;
    }
    // Most prefix probes stay in one linear region of the clipped sequence.
    // Use its arithmetic sum directly; only probes crossing an edge divide.
    if (first >= 0 && last <= limit) {
        return count * (first + last) / 2;
    }
    const int64_t positive = first < 1 ? (1 - first + step - 1) / step : 0;
    const int64_t saturated = first < limit ? (limit - first + step - 1) / step : 0;
    const int64_t begin = positive < count ? positive : count;
    const int64_t end = saturated < count ? saturated : count;
    const int64_t middle = end - begin;
    return middle * (2 * first + (begin + end - 1) * step) / 2 + (count - end) * limit;
}

inline int64_t GqaBandColumnOwnerSequencePrefix(int64_t m, int64_t n, int64_t p, int64_t q, int64_t first, int64_t step,
                                                int64_t pairs)
{
    if (first > n || pairs <= 0) {
        return 0;
    }
    const int64_t count = first + (pairs - 1) * step <= n ? pairs : (n - first) / step + 1;
    return GqaBandClippedProgressionSum(first + p - 1, step, count, m) -
           GqaBandClippedProgressionSum(first - q, step, count, m);
}

inline int64_t GqaBandColumnOwnerPairPrefix(int64_t m, int64_t n, int64_t p, int64_t q, int64_t lanes, int64_t lane,
                                            int64_t pairs)
{
    return GqaBandColumnOwnerSequencePrefix(m, n, p, q, lane + 1, 2 * lanes, pairs) +
           GqaBandColumnOwnerSequencePrefix(m, n, p, q, 2 * lanes - lane, 2 * lanes, pairs);
}

// Positive BAND column heights form a clipped trapezoid. In descending-height
// order there is one plateau, then paired edges, then at most one longer edge.
// This representation depends only on geometry and uses no per-column table.
inline int64_t GqaBandBalancedColumnCount(int64_t m, int64_t n, int64_t q)
{
    return n < m + q - 1 ? n : m + q - 1;
}

inline int64_t GqaBandBalancedSequencePrefix(int64_t m, int64_t n, int64_t p, int64_t q, int64_t first, int64_t step,
                                             int64_t count)
{
    const int64_t columns = GqaBandBalancedColumnCount(m, n, q);
    if (first >= columns || count <= 0) {
        return 0;
    }
    const int64_t available = (columns - 1 - first) / step + 1;
    count = count < available ? count : available;
    const int64_t height = m < p + q - 1 ? m : p + q - 1;
    const int64_t peak = height < n + p - 1 ? height : n + p - 1;
    const int64_t left = GqaBandMax(1, peak - p + 1);
    const int64_t right = columns < m + q - peak ? columns : m + q - peak;
    const int64_t plateau = right - left + 1;
    const int64_t paired = left - 1 < columns - right ? left - 1 : columns - right;
    // step=2*lanes is even. At paired ranks the height drops once per two
    // ranks, so each progression drops by exactly step/2 per element.
    const int64_t diff = first - plateau;
    const int64_t firstLevel = diff >= 0 ? diff / 2 + 1 : -((-diff - 1) / 2);
    const int64_t pairedDrop = paired > 0 ? GqaBandClippedProgressionSum(firstLevel, step / 2, count, paired) : 0;
    const int64_t single = columns - plateau - 2 * paired;
    const int64_t singleDrop =
        single > 0 ? GqaBandClippedProgressionSum(first - plateau - 2 * paired + 1, step, count, single) : 0;
    return count * peak - pairedDrop - singleDrop;
}

inline int64_t GqaBandBalancedPairPrefix(int64_t m, int64_t n, int64_t p, int64_t q, int64_t lanes, int64_t lane,
                                         int64_t pairs)
{
    return GqaBandBalancedSequencePrefix(m, n, p, q, lane, 2 * lanes, pairs) +
           GqaBandBalancedSequencePrefix(m, n, p, q, 2 * lanes - 1 - lane, 2 * lanes, pairs);
}

inline int64_t GqaBandBalancedPhysicalColumn(int64_t m, int64_t n, int64_t p, int64_t q, int64_t rank)
{
    const int64_t columns = GqaBandBalancedColumnCount(m, n, q);
    const int64_t height = m < p + q - 1 ? m : p + q - 1;
    const int64_t peak = height < n + p - 1 ? height : n + p - 1;
    const int64_t left = GqaBandMax(1, peak - p + 1);
    const int64_t right = columns < m + q - peak ? columns : m + q - peak;
    const int64_t plateau = right - left + 1;
    const int64_t paired = left - 1 < columns - right ? left - 1 : columns - right;
    if (rank < plateau) {
        return left + rank;
    }
    rank -= plateau;
    if (rank < 2 * paired) {
        const int64_t offset = rank / 2 + 1;
        return rank % 2 == 0 ? left - offset : right + offset;
    }
    const int64_t offset = rank - paired + 1;
    return left - 1 > columns - right ? left - offset : right + offset;
}

inline int64_t GqaBandOwnerPairPrefix(int64_t m, int64_t n, int64_t p, int64_t q, int64_t lanes, int64_t lane,
                                      int64_t pairs, bool balanced)
{
    return balanced ? GqaBandBalancedPairPrefix(m, n, p, q, lanes, lane, pairs) :
                      GqaBandColumnOwnerPairPrefix(m, n, p, q, lanes, lane, pairs);
}

template <bool Balanced>
inline void GqaBandColumnOwnerIndexImpl(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p, int64_t q,
                                        int64_t core, int64_t round, int64_t& w, int64_t& x, int64_t& y)
{
    w = -1;
    if (!CanUseGqaBandColumnOwner(k, b, g, m, n, p, q) || core < 1 || core > k || round < 1) {
        return;
    }
    const int64_t lanes = GqaBandColumnOwnerLanes(k, b, g);
    if (core > b * lanes) {
        return;
    }
    const int64_t lane = (core - 1) % lanes;
    const int64_t group = (core - 1) / lanes;
    const int64_t rowSlot = (round - 1) / g;
    const int64_t columns = Balanced ? GqaBandBalancedColumnCount(m, n, q) : n;
    const int64_t pairCount = (columns + 2 * lanes - 1) / (2 * lanes);
    if (rowSlot >= GqaBandOwnerPairPrefix(m, n, p, q, lanes, lane, pairCount, Balanced)) {
        return;
    }
    // Find the first column pair whose inclusive prefix exceeds rowSlot.
    // Empty columns and an incomplete final pair need no shape-specific path.
    int64_t lo = 0;
    int64_t hi = pairCount;
    while (lo + 1 < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (GqaBandOwnerPairPrefix(m, n, p, q, lanes, lane, mid, Balanced) <= rowSlot) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    int64_t localRow = rowSlot - GqaBandOwnerPairPrefix(m, n, p, q, lanes, lane, lo, Balanced);
    const int64_t firstRank = 2 * lanes * lo + lane;
    y = Balanced ? GqaBandBalancedPhysicalColumn(m, n, p, q, firstRank) : firstRank + 1;
    const int64_t last = y + p - 1 < m ? y + p - 1 : m;
    const int64_t first = GqaBandMax(1, y - q + 1);
    const int64_t height = y <= n ? GqaBandMax(0, last - first + 1) : 0;
    if (localRow >= height) {
        localRow -= height;
        const int64_t secondRank = 2 * lanes * lo + 2 * lanes - 1 - lane;
        y = Balanced ? GqaBandBalancedPhysicalColumn(m, n, p, q, secondRank) : secondRank + 1;
    }
    x = GqaBandMax(1, y - q + 1) + localRow;
    // lanes <= g gives distinct query heads within a group in every round.
    // Unique column ownership protects dK/dV. Across g rounds each row visits
    // every query head once, while all rows of a column remain consecutive.
    w = group * g + ((round - 1 + lane) % g) + 1;
}

// Keep the new rank permutation out of the common scheduling instruction path.
// The original column order remains a compile-time specialization.
inline void GqaBandBalancedColumnOwnerIndex(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p, int64_t q,
                                            int64_t core, int64_t round, int64_t& w, int64_t& x, int64_t& y)
{
    GqaBandColumnOwnerIndexImpl<true>(k, b, g, m, n, p, q, core, round, w, x, y);
}

inline void GqaBandColumnOwnerIndex(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p, int64_t q,
                                    int64_t core, int64_t round, int64_t& w, int64_t& x, int64_t& y,
                                    bool balanced = false)
{
    if (balanced) {
        GqaBandBalancedColumnOwnerIndex(k, b, g, m, n, p, q, core, round, w, x, y);
        return;
    }

    w = -1;
    if (!CanUseGqaBandColumnOwner(k, b, g, m, n, p, q) || core < 1 || core > k || round < 1) {
        return;
    }
    const int64_t lanes = GqaBandColumnOwnerLanes(k, b, g);
    if (core > b * lanes) {
        return;
    }
    const int64_t lane = (core - 1) % lanes;
    const int64_t group = (core - 1) / lanes;
    const int64_t rowSlot = (round - 1) / g;
    // At most one mirrored column pair per lane: evaluate its two intervals
    // directly instead of forming clipped progression prefixes. The selected
    // tasks, head order and empty slots are identical to the general mapper.
    if (n <= 2 * lanes) {
        int64_t localRow = rowSlot;
        y = lane + 1;
        int64_t first = GqaBandMax(1, y - q + 1);
        const int64_t last = y + p - 1 < m ? y + p - 1 : m;
        const int64_t height = y <= n ? GqaBandMax(0, last - first + 1) : 0;
        if (localRow >= height) {
            localRow -= height;
            y = 2 * lanes - lane;
            first = GqaBandMax(1, y - q + 1);
            const int64_t secondLast = y + p - 1 < m ? y + p - 1 : m;
            if (y > n || localRow >= GqaBandMax(0, secondLast - first + 1)) {
                return;
            }
        }
        // For a flat lower boundary, visit the first column from its common
        // last row downward. Keep the second column in its original order.
        // Heads, column owners, continuous KV runs and round counts are unchanged.
        x = p == m && rowSlot < height ? m - localRow : first + localRow;
        w = group * g + ((round - 1 + lane) % g) + 1;
        return;
    }
    const int64_t pairCount = (n + 2 * lanes - 1) / (2 * lanes);
    if (rowSlot >= GqaBandColumnOwnerPairPrefix(m, n, p, q, lanes, lane, pairCount)) {
        return;
    }
    // Find the first column pair whose inclusive prefix exceeds rowSlot.
    // Empty columns and an incomplete final pair need no shape-specific path.
    int64_t lo = 0;
    int64_t hi = pairCount;
    while (lo + 1 < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (GqaBandColumnOwnerPairPrefix(m, n, p, q, lanes, lane, mid) <= rowSlot) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    int64_t localRow = rowSlot - GqaBandColumnOwnerPairPrefix(m, n, p, q, lanes, lane, lo);
    y = 2 * lanes * lo + lane + 1;
    const int64_t last = y + p - 1 < m ? y + p - 1 : m;
    const int64_t first = GqaBandMax(1, y - q + 1);
    const int64_t height = y <= n ? GqaBandMax(0, last - first + 1) : 0;
    if (localRow >= height) {
        localRow -= height;
        y = 2 * lanes * lo + 2 * lanes - lane;
    }
    x = GqaBandMax(1, y - q + 1) + localRow;
    // lanes <= g gives distinct query heads within a group in every round.
    // Unique column ownership protects dK/dV. Across g rounds each row visits
    // every query head once, while all rows of a column remain consecutive.
    w = group * g + ((round - 1 + lane) % g) + 1;
}

// One-based head-inner -> head-major round in each independent headSpan * g block.
// All cores must use the same permutation: never reorder an individual core's tasks.
inline int64_t GqaBandHeadInnerRound(int64_t r, int64_t headSpan, int64_t g, int64_t shift)
{
    if (r <= 0 || headSpan <= 0 || g <= 0) {
        return r;
    }
    const int64_t period = headSpan * g;
    const int64_t local = (r - 1) % period;
    return (r - 1) / period * period + (local % g) * headSpan + (local / g + shift) % headSpan + 1;
}

inline bool IsGqaBandEdgePair(const GqaBandPhase& phase)
{
    return phase.kind == GqaBandPhaseKind::EDGE_PAIR || phase.kind == GqaBandPhaseKind::EDGE_PAIR_Q_BOUND ||
           phase.kind == GqaBandPhaseKind::EDGE_PAIR_P_BOUND;
}

// The lower part of a paired virtual column is selected by x > split + y.
inline int64_t GqaBandEdgePairSplit(const GqaBandPhase& phase)
{
    if (phase.kind == GqaBandPhaseKind::EDGE_PAIR) {
        return phase.n - 1;
    }
    if (phase.kind == GqaBandPhaseKind::EDGE_PAIR_Q_BOUND) {
        return phase.colOffset - phase.rowOffset - phase.n;
    }
    if (phase.kind == GqaBandPhaseKind::EDGE_PAIR_P_BOUND) {
        return phase.m + phase.rowOffset - phase.colOffset - 1;
    }
    return -1;
}

inline bool CanReorderGqaBandEdgePair(const GqaBandPhase& phase, int64_t k, int64_t groups, int64_t g)
{
    if (!IsGqaBandEdgePair(phase) || k <= 0 || groups <= 0 || g <= 1 || phase.m <= 0 || phase.n <= 0 ||
        phase.n > phase.m || groups > k / phase.n) {
        return false;
    }
    // Here active cores == groups*n and denseR == g. Each core owns one complete
    // virtual KV column, its row offset is y, and the dense ID correction is off.
    const int64_t split = GqaBandEdgePairSplit(phase);
    return split >= 0 && split < phase.m - phase.n;
}

inline int64_t GqaBandEdgePairSourceRound(const GqaBandPhase& phase, int64_t k, int64_t groups, int64_t g, int64_t r)
{
    if (!CanReorderGqaBandEdgePair(phase, k, groups, g) || r <= 0 || (r - 1) / phase.m >= g) {
        return r;
    }
    // All cores use the same bijection of the m*g phase rounds. Starting just
    // below x=split+y makes each of the two physical KV columns one continuous run.
    return GqaBandHeadInnerRound(r, phase.m, g, GqaBandEdgePairSplit(phase) + 1);
}

// Square-triangle specialization of CalGQACausalIndex's rm1/rm2/rm3/rm4 boundaries.
// Only the source round changes; phase lengths, per-core work and same-round address
// uniqueness remain unchanged. The caller bounds r by the full/tail phase rounds.
inline int64_t GqaBandTriangleSourceRound(int64_t k, int64_t size, int64_t b, int64_t g, int64_t r)
{
    if (k <= 0 || size <= 0 || b <= 0 || g <= 1 || r <= 0 || (b < k && size < k)) {
        return r;
    }
    const int64_t triangleSize = size * (size + 1) / 2;
    const int64_t fullRounds = (b / k) * triangleSize * g;
    if (r <= fullRounds) {
        return GqaBandHeadInnerRound(r, triangleSize, g, 0);
    }

    const int64_t tail = b % k;
    const int64_t t = size / k;
    const int64_t height = 2 * size - t * k + 1;
    const int64_t pairedRounds = height * t * (tail / 2) * g;
    int64_t localRound = r - fullRounds;
    if (localRound <= pairedRounds) {
        // For t == 1, a common cyclic shift aligns the two physical KV-column runs.
        // For t > 1, keep the existing spatial start and only transpose the heads.
        const int64_t shift = t == 1 ? size - k + 1 : 0;
        return fullRounds + GqaBandHeadInnerRound(localRound, height, g, shift);
    }

    localRound -= pairedRounds;
    int64_t oddRounds = 0;
    if (tail % 2 != 0) {
        if (t % 2 != 0) {
            const int64_t t1 = size / (2 * k);
            const int64_t m1 = size - t1 * 2 * k;
            oddRounds = (size + m1 + 1) * t1;
            oddRounds += size % k == 0 ? m1 : GqaBandMax(m1, 2 * m1 - 2 * k + 1);
        } else {
            oddRounds = height * t / 2;
        }
    }
    if (localRound <= oddRounds * g) {
        return fullRounds + pairedRounds + GqaBandHeadInnerRound(localRound, oddRounds, g, 0);
    }
    // The ell remainder already interleaves Q heads; do not transpose it again.
    return r;
}
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_BAND_INDEX_H_
