/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_BAND_CANDIDATES_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_BAND_CANDIDATES_H_

// BAND candidate construction, cost evaluation and cache; no selection priority.
#include <algorithm>
#include <vector>
#include "index.h"
#include "schedule_types.h"
#include "../fag_deter_utils.h"

namespace optiling {
namespace fag_pto_arch35 {
enum class GqaBandGeometryMode : uint8_t {
    GENERAL,
    UPPER_TRIANGLE,
    REVERSED_GENERAL,
    REVERSED_UPPER_TRIANGLE,
    PAIRED_EDGES,
    GENERAL_PAIRED_EDGES
};

struct GqaBandPlan {
    GqaBandPhase phases[GQA_BAND_MAX_PHASES];
    GqaBandPhase tailPhases[GQA_BAND_MAX_PHASES];
    GqaBandGeometryMode geometryMode = GqaBandGeometryMode::GENERAL;
    GqaBandGeometryMode tailGeometryMode = GqaBandGeometryMode::GENERAL;
    int64_t phaseCount = 0;
    // Zero means that the incomplete chunk reuses phases and their tailRounds.
    int64_t tailPhaseCount = 0;
    int64_t k = 0;
    int64_t b = 0;
    int64_t g = 0;
    int64_t groupsPerChunk = 0;
    int64_t fullRounds = 0;
    int64_t tailRounds = 0;
    int64_t totalRounds = 0;
};

// Copy only active phases, so failed/reused candidates cannot leak stale metadata.
inline void SaveGqaBandSchedule(const GqaBandPlan& plan, GqaBandSchedule& schedule)
{
    schedule = GqaBandSchedule{};
    schedule.phaseCount = plan.phaseCount;
    schedule.tailPhaseCount = plan.tailPhaseCount;
    schedule.k = plan.k;
    schedule.b = plan.b;
    schedule.g = plan.g;
    schedule.groupsPerChunk = plan.groupsPerChunk;
    schedule.fullRounds = plan.fullRounds;
    schedule.tailRounds = plan.tailRounds;
    schedule.totalRounds = plan.totalRounds;
    for (int64_t i = 0; i < plan.phaseCount; ++i) {
        schedule.phases[i] = plan.phases[i];
    }
    for (int64_t i = 0; i < plan.tailPhaseCount; ++i) {
        schedule.tailPhases[i] = plan.tailPhases[i];
    }
}

inline void AppendGqaBandPhase(GqaBandPlan& plan, GqaBandPhaseKind kind, int64_t m, int64_t n, int64_t rowOffset,
                               int64_t colOffset)
{
    if (m <= 0 || n <= 0) {
        return;
    }
    if (plan.phaseCount > 0 && kind == GqaBandPhaseKind::RECTANGLE) {
        GqaBandPhase& previous = plan.phases[plan.phaseCount - 1];
        if (previous.kind == kind && previous.m == m && previous.rowOffset == rowOffset &&
            previous.colOffset + previous.n == colOffset) {
            previous.n += n;
            return;
        }
    }
    GqaBandPhase& phase = plan.phases[plan.phaseCount++];
    phase.kind = kind;
    phase.m = m;
    phase.n = n;
    phase.rowOffset = rowOffset;
    phase.colOffset = colOffset;
}

// Encode a proved folded domain without changing the established phase tags.
inline void AppendGqaBandFoldedPhase(GqaBandPlan& plan, GqaBandPhaseKind kind, const GqaBandDomain& domain)
{
    if (kind == GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER) {
        AppendGqaBandPhase(plan, kind, domain.m, domain.n, domain.p, domain.q);
    } else if (kind == GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER) {
        AppendGqaBandPhase(plan, kind, domain.m, domain.n, domain.q, domain.colOrigin);
    } else {
        AppendGqaBandPhase(plan, kind, domain.m, domain.n, domain.rowOrigin, domain.p);
    }
}

// One-based valid rows of column y: [max(1,y-q+1), min(m,y+p-1)].
// Positive p/q only. There are at most three column intervals and five phases.
inline void BuildGqaBandGeneralGeometry(int64_t m, int64_t n, int64_t p, int64_t q, GqaBandPlan& plan)
{
    plan.phaseCount = 0;
    plan.geometryMode = GqaBandGeometryMode::GENERAL;
    const int64_t end = GqaBandMin(n, m + q - 1);
    int64_t first = 1;
    while (first <= end) {
        int64_t next = end + 1;
        if (q > first) {
            next = GqaBandMin(next, q);
        }
        if (m - p + 1 > first) {
            next = GqaBandMin(next, m - p + 1);
        }
        const int64_t last = next - 1;
        const int64_t width = next - first;
        const int64_t lo = GqaBandMax(1, first - q + 1);
        const int64_t hi = GqaBandMin(m, first + p - 1);
        const int64_t loLast = GqaBandMax(1, last - q + 1);
        const int64_t hiLast = GqaBandMin(m, last + p - 1);
        const int64_t height = hi - lo + 1;
        if (lo == loLast && hi == hiLast) {
            AppendGqaBandPhase(plan, GqaBandPhaseKind::RECTANGLE, height, width, lo - 1, first - 1);
        } else if (lo == loLast) {
            AppendGqaBandPhase(plan, GqaBandPhaseKind::RECTANGLE, height - 1, width, lo - 1, first - 1);
            AppendGqaBandPhase(plan, GqaBandPhaseKind::UPPER_TRIANGLE, width, width, lo + height - 2, first - 1);
        } else if (hi == hiLast) {
            AppendGqaBandPhase(plan, GqaBandPhaseKind::LOWER_TRIANGLE, width, width, lo - 1, first - 1);
            AppendGqaBandPhase(plan, GqaBandPhaseKind::RECTANGLE, height - width, width, lo + width - 1, first - 1);
        } else {
            AppendGqaBandPhase(plan, GqaBandPhaseKind::DIAGONAL, height, width, lo - 1, first - 1);
        }
        first = next;
    }
}

// Preserve the original piecewise geometry as the tie-breaking candidate.
inline void BuildGqaBandGeometry(int64_t m, int64_t n, int64_t p, int64_t q, GqaBandPlan& plan)
{
    if (p != 1 || q != n) {
        BuildGqaBandGeneralGeometry(m, n, p, q, plan);
        return;
    }
    plan.phaseCount = 0;
    plan.geometryMode = GqaBandGeometryMode::UPPER_TRIANGLE;
    const int64_t size = GqaBandMin(m, n);
    AppendGqaBandPhase(plan, GqaBandPhaseKind::UPPER_TRIANGLE, size, size, 0, 0);
    AppendGqaBandPhase(plan, GqaBandPhaseKind::RECTANGLE, m, n - size, 0, size);
}

inline int64_t GqaBandDenseRounds(int64_t k, int64_t m, int64_t n, int64_t b, int64_t g)
{
    k = GqaBandMin(k, GqaBandMin(b * g * m, b * n));
    return m * GqaBandMax(g, GqaBandMax(GqaBandCeil(b * n * g, k), GqaBandCeil(n, m)));
}

inline int64_t GqaBandOverlapChunkRounds(int64_t k, int64_t groups, int64_t g, int64_t m, int64_t n)
{
    if (!CanUseGqaBandOverlap(k, groups, g, m, n)) {
        return 0;
    }
    int64_t suffixRounds = 0;
    for (int64_t group = 0; group < groups; ++group) {
        const int64_t localK = GqaBandOverlapCoreBudget(k, groups, m, n, group);
        const int64_t suffixWidth = n - (localK - m / 2);
        if (suffixWidth > 0) {
            suffixRounds = GqaBandMax(suffixRounds, GqaBandDenseRounds(localK, m, suffixWidth, 1, g));
        }
    }
    return m * g + suffixRounds;
}

// CalcCausalDeterRound specialized to a square triangle, G > 1.
inline int64_t GqaBandTriangleRounds(int64_t k, int64_t size, int64_t b, int64_t g)
{
    int64_t tail = b % k;
    const int64_t t = size / k;
    const int64_t ell = size % k;
    const int64_t t1 = size / (2 * k);
    const int64_t n1 = t * k;
    int64_t rounds = b / k * size * (size + 1) / 2;
    rounds += (2 * size - n1 + 1) * t * (tail / 2);
    if (tail % 2 != 0) {
        if (t % 2 != 0) {
            const int64_t m1 = size - t1 * 2 * k;
            rounds += (size + m1 + 1) * t1;
            rounds += ell == 0 ? m1 : GqaBandMax(m1, 2 * m1 - 2 * k + 1);
            --tail;
        } else {
            rounds += (2 * size - n1 + 1) * t / 2;
        }
    }
    rounds += GqaBandCeil(((ell + 1) / 2) * tail, k) * (ell + (ell % 2 == 0 ? 1 : 0));
    return rounds * g;
}

inline int64_t GqaBandColumnOwnerRounds(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p, int64_t q,
                                        bool balanced = false)
{
    if (!CanUseGqaBandColumnOwner(k, b, g, m, n, p, q)) {
        return 0;
    }
    const int64_t lanes = GqaBandColumnOwnerLanes(k, b, g);
    const int64_t columns = balanced ? GqaBandBalancedColumnCount(m, n, q) : n;
    const int64_t pairs = (columns + 2 * lanes - 1) / (2 * lanes);
    int64_t maxRows = 0;
    for (int64_t lane = 0; lane < lanes; ++lane) {
        maxRows = GqaBandMax(maxRows, GqaBandOwnerPairPrefix(m, n, p, q, lanes, lane, pairs, balanced));
    }
    return maxRows * g;
}

inline int64_t GqaBandFoldedColumnOwnerRounds(int64_t k, int64_t b, int64_t g, int64_t m, int64_t n, int64_t p,
                                              int64_t q)
{
    if (!CanUseGqaBandFoldedColumnOwner(k, b, g, m, n, p, q)) {
        return 0;
    }
    return GqaBandCeil(b * GqaBandMin(m, n), k) * (p + q - 1) * g;
}

inline int64_t GqaBandPhaseRounds(const GqaBandPhase& phase, int64_t k, int64_t b, int64_t g)
{
    if (b == 0) {
        return 0;
    }
    if (phase.kind == GqaBandPhaseKind::OVERLAP_LOWER || phase.kind == GqaBandPhaseKind::OVERLAP_UPPER) {
        return GqaBandOverlapChunkRounds(k, b, g, phase.m, phase.n);
    }
    if (phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE || phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE) {
        return GqaBandTriangleRounds(k, phase.m, b, g);
    }
    if (phase.kind == GqaBandPhaseKind::DIAGONAL_COLUMN_OWNER) {
        return GqaBandCeil(b * phase.n, k) * g;
    }
    if (phase.kind == GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER ||
        phase.kind == GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER ||
        phase.kind == GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER) {
        const GqaBandDomain domain = DecodeGqaBandFoldedDomain(phase);
        return GqaBandFoldedColumnOwnerRounds(k, b, g, domain.m, domain.n, domain.p, domain.q);
    }
    if (phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER) {
        return phase.m == phase.n ? GqaBandColumnOwnerRounds(k, b, g, phase.m, phase.n, phase.m, 1) : 0;
    }
    if (phase.kind == GqaBandPhaseKind::BAND_COLUMN_OWNER ||
        phase.kind == GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER) {
        return GqaBandColumnOwnerRounds(k, b, g, phase.m, phase.n, phase.rowOffset, phase.colOffset,
                                        phase.kind == GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER);
    }
    return GqaBandDenseRounds(k, phase.m, phase.n, b, g);
}

inline void SetGqaBandGroups(int64_t k, int64_t b, int64_t g, int64_t groups, GqaBandPlan& plan)
{
    plan.k = k;
    plan.b = b;
    plan.g = g;
    plan.groupsPerChunk = groups;
    plan.tailPhaseCount = 0;
    plan.tailGeometryMode = GqaBandGeometryMode::GENERAL;
    plan.fullRounds = 0;
    plan.tailRounds = 0;
    for (int64_t i = 0; i < plan.phaseCount; ++i) {
        GqaBandPhase& phase = plan.phases[i];
        phase.fullRounds = GqaBandPhaseRounds(phase, k, groups, g);
        phase.tailRounds = GqaBandPhaseRounds(phase, k, b % groups, g);
        plan.fullRounds += phase.fullRounds;
        plan.tailRounds += phase.tailRounds;
    }
    plan.totalRounds = (b / groups) * plan.fullRounds + plan.tailRounds;
}

// Normalize either complete-domain orientation to the same rectangle/triangle
// mapper. No condition depends on D, a shape list or an external case identifier.
inline bool BuildGqaBandOverlapCandidate(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g,
                                         int64_t groups, GqaBandPlan& plan)
{
    plan = GqaBandPlan{};
    if (b <= 0 || groups > b || p <= 0 || p > m || q <= 0 || q > n) {
        return false;
    }
    GqaBandPhaseKind kind;
    int64_t rectangleWidth = 0;
    if (p == m && n - q >= m - 1) {
        kind = GqaBandPhaseKind::OVERLAP_LOWER;
        rectangleWidth = q;
    } else if (p == 1 && q == n && n >= m) {
        kind = GqaBandPhaseKind::OVERLAP_UPPER;
        rectangleWidth = n - m + 1;
    } else {
        return false;
    }
    if (!CanUseGqaBandOverlap(k, groups, g, m, rectangleWidth) ||
        (b % groups != 0 && !CanUseGqaBandOverlap(k, b % groups, g, m, rectangleWidth))) {
        return false;
    }
    AppendGqaBandPhase(plan, kind, m, rectangleWidth, 0, 0);
    SetGqaBandGroups(k, b, g, groups, plan);
    return plan.fullRounds > 0 && (b % groups == 0 || plan.tailRounds > 0);
}

// Reflect phase domains, not dQ/dKV axes. Each phase still uses its existing
// conflict-free local mapper; no extra coordinate transform is needed per round.
inline void ReflectGqaBandGeometry(int64_t m, int64_t n, GqaBandPlan& plan)
{
    for (int64_t i = 0; i < plan.phaseCount; ++i) {
        GqaBandPhase& phase = plan.phases[i];
        const int64_t rowSpan = phase.kind == GqaBandPhaseKind::DIAGONAL ? phase.m + phase.n - 1 : phase.m;
        phase.rowOffset = m - phase.rowOffset - rowSpan;
        phase.colOffset = n - phase.colOffset - phase.n;
        if (phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE) {
            phase.kind = GqaBandPhaseKind::UPPER_TRIANGLE;
        } else if (phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE) {
            phase.kind = GqaBandPhaseKind::LOWER_TRIANGLE;
        }
    }
    plan.geometryMode = plan.geometryMode == GqaBandGeometryMode::UPPER_TRIANGLE ?
                            GqaBandGeometryMode::REVERSED_UPPER_TRIANGLE :
                            GqaBandGeometryMode::REVERSED_GENERAL;
}

// Build one admissible complete geometry without evaluating other candidates.
inline bool BuildGqaBandGeometryByMode(int64_t m, int64_t n, int64_t p, int64_t q, GqaBandGeometryMode mode,
                                       GqaBandPlan& plan)
{
    if (mode == GqaBandGeometryMode::GENERAL_PAIRED_EDGES) {
        // Keep the previous symmetric candidate and its reduction order unchanged.
        if (p <= 0 || q <= 0 || (m == n && p == q)) {
            return false;
        }
        const int64_t validRows = GqaBandMin(m, n + p - 1);
        const int64_t validCols = GqaBandMin(n, m + q - 1);
        const int64_t pairWidth = GqaBandMin(q, validCols + p - validRows);
        // Both physical row domains and column domains must be disjoint.
        if (pairWidth <= 0 || validCols < 2 * pairWidth || validCols < p + q + 2 * pairWidth - 2) {
            return false;
        }
        const int64_t middleWidth = validCols - 2 * pairWidth;
        plan.phaseCount = 0;
        if (middleWidth > 0) {
            const int64_t firstRow = GqaBandMax(1, pairWidth - q + 2);
            const int64_t lastRow = GqaBandMin(validRows, validCols - pairWidth + p - 1);
            BuildGqaBandGeneralGeometry(lastRow - firstRow + 1, middleWidth, p + pairWidth - firstRow + 1,
                                        q - pairWidth + firstRow - 1, plan);
            // Removing a complete edge slope leaves at most four middle phases.
            // Insert the paired phase first without adding another plan on the stack.
            for (int64_t i = plan.phaseCount; i > 0; --i) {
                plan.phases[i] = plan.phases[i - 1];
                plan.phases[i].rowOffset += firstRow - 1;
                plan.phases[i].colOffset += pairWidth;
            }
        }
        const int64_t middlePhaseCount = plan.phaseCount;
        plan.phaseCount = 0;
        const GqaBandPhaseKind pairKind =
            pairWidth == q ? GqaBandPhaseKind::EDGE_PAIR_Q_BOUND : GqaBandPhaseKind::EDGE_PAIR_P_BOUND;
        AppendGqaBandPhase(plan, pairKind, validRows - validCols + p + q + pairWidth - 1, pairWidth,
                           validCols - pairWidth - p - q + 1, validCols - pairWidth);
        plan.phaseCount += middlePhaseCount;
        plan.geometryMode = mode;
        return true;
    }
    if (mode == GqaBandGeometryMode::PAIRED_EDGES) {
        // Pair the two edge trapezoids only when both physical row domains are
        // disjoint. Column disjointness alone does not preserve dQ uniqueness.
        if (m != n || p != q || p <= 0 || m < 2 * p || m < 4 * p - 2) {
            return false;
        }
        plan.phaseCount = 0;
        plan.geometryMode = mode;
        // EDGE_PAIR offsets apply only to the lower part of the virtual rectangle.
        AppendGqaBandPhase(plan, GqaBandPhaseKind::EDGE_PAIR, 3 * p - 1, p, m - 3 * p + 1, m - p);
        AppendGqaBandPhase(plan, GqaBandPhaseKind::DIAGONAL, 2 * p - 1, m - 2 * p, 1, p);
        return true;
    }
    const bool reversed =
        mode == GqaBandGeometryMode::REVERSED_GENERAL || mode == GqaBandGeometryMode::REVERSED_UPPER_TRIANGLE;
    const bool upper =
        mode == GqaBandGeometryMode::UPPER_TRIANGLE || mode == GqaBandGeometryMode::REVERSED_UPPER_TRIANGLE;
    if (reversed) {
        const int64_t reflectedP = m - n + q;
        const int64_t reflectedQ = n - m + p;
        if (reflectedP <= 0 || reflectedQ <= 0 || (reflectedP == p && reflectedQ == q)) {
            return false;
        }
        p = reflectedP;
        q = reflectedQ;
    }
    if (upper) {
        if (p != 1 || q != n) {
            return false;
        }
        BuildGqaBandGeometry(m, n, p, q, plan);
    } else {
        BuildGqaBandGeneralGeometry(m, n, p, q, plan);
    }
    if (reversed) {
        ReflectGqaBandGeometry(m, n, plan);
    }
    return true;
}

// For n=s*k, m>=k and s>1, serializing KV groups preserves every column's
// core owner ceil(y/s) and the exact round count. Each group's phase interval
// becomes a subinterval of its old interval, including at full/tail boundaries.
inline void ConfigureGqaBandDensePhases(GqaBandPlan& plan)
{
    if (plan.k <= 0 || plan.groupsPerChunk <= 0) {
        return;
    }
    for (int64_t i = 0; i < plan.phaseCount + plan.tailPhaseCount; ++i) {
        const bool isTail = i >= plan.phaseCount;
        GqaBandPhase& phase = isTail ? plan.tailPhases[i - plan.phaseCount] : plan.phases[i];
        if (phase.kind != GqaBandPhaseKind::RECTANGLE && phase.kind != GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR) {
            continue;
        }
        const int64_t groups = isTail ? plan.b % plan.groupsPerChunk : plan.groupsPerChunk;
        const bool batchMajor = groups > 1 && phase.m >= plan.k && phase.n > plan.k && phase.n % plan.k == 0;
        phase.kind = batchMajor ? GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR : GqaBandPhaseKind::RECTANGLE;
    }
}

// Match GenGQABandInfo, including its trailing rm2 when b % k == 0.
// The old host deterMaxRound is NOT the GQA kernel's round bound.
inline int64_t GqaBandLegacyRounds(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g)
{
    k = GqaBandMin(k, GqaBandMin(b * g * m, b * n));
    const int64_t tail = b % k;
    if (p + q > m) {
        n = q + GqaBandMin(m - 1, n - q);
    } else {
        const int64_t middle = GqaBandMin(n - q + 1, m + 2 - p - q);
        const int64_t last = GqaBandMax(0, GqaBandMin(p + n - m - 1, p + q - 2));
        if (last == 0) {
            m = p + q + middle - 2;
        }
        n = q - 1 + middle + last;
    }
    const int64_t area = m * n - (m - p) * (m - p + 1) / 2 - (n - q) * (n - q + 1) / 2;
    int64_t remaining;
    if (p + q > m) {
        const int64_t columns = GqaBandMin(n, p + q - 1);
        remaining = m * GqaBandMax(g, GqaBandMax(GqaBandCeil(tail * columns * g, k), GqaBandCeil(columns, m)));
    } else {
        const int64_t height = p + q - 1;
        remaining = height * GqaBandMax(g, GqaBandMax(GqaBandCeil(tail * n * g, k), GqaBandCeil(n, height)));
        if (n > m) {
            const int64_t alternate =
                m * GqaBandMax(g, GqaBandMax(GqaBandCeil(tail * height * g, k), GqaBandCeil(height, m)));
            remaining = GqaBandMin(remaining, alternate);
        }
    }
    return (b / k) * area * g + remaining;
}

// Host-only locality admission. These counts describe logical KV loads/dKV
// submissions, not DRAM traffic or measured latency. The selector bounds all
// products before calling the planner; no loop here depends on sequence length.

inline int64_t GqaBandDenseKvRunUpperBound(int64_t k, int64_t m, int64_t n, int64_t groups, int64_t g, bool noTune)
{
    if (k <= 0 || m <= 0 || n <= 0 || groups <= 0 || g <= 0) {
        return -1;
    }
    const int64_t denseRound = GqaBandDenseRounds(k, m, n, groups, g) / m;
    const int64_t headColumns = groups * n * g;
    if (!noTune && denseRound / GqaBandGcd(groups * g, denseRound) * m < n) {
        // ID correction can permute head-columns, but all m rows of one ID
        // still use one physical KV column. Do not assume adjacent IDs merge.
        return headColumns;
    }
    // Each core owns a contiguous ID interval. A core boundary inside a group
    // of g head IDs splits one KV run; a boundary divisible by g does not.
    const int64_t boundaries = GqaBandCeil(headColumns, denseRound) - 1;
    const int64_t alignedPeriod = g / GqaBandGcd(denseRound, g);
    return groups * n + boundaries - boundaries / alignedPeriod;
}

inline int64_t GqaBandOverlapChunkKvRunUpperBound(int64_t k, int64_t groups, int64_t g, int64_t m, int64_t n)
{
    if (!CanUseGqaBandOverlap(k, groups, g, m, n)) {
        return -1;
    }
    int64_t runs = 0;
    for (int64_t group = 0; group < groups; ++group) {
        const int64_t localK = GqaBandOverlapCoreBudget(k, groups, m, n, group);
        const int64_t stripeWidth = localK - m / 2;
        const int64_t suffixWidth = n - stripeWidth;
        // Triangle, stripe and suffix use disjoint physical KV columns.
        runs += m - 1 + stripeWidth;
        if (suffixWidth > 0) {
            runs += GqaBandDenseKvRunUpperBound(localK, m, suffixWidth, 1, g, false);
        }
    }
    return runs;
}

inline int64_t GqaBandPhaseKvRunUpperBound(const GqaBandPhase& phase, int64_t k, int64_t groups, int64_t g)
{
    if (k <= 0 || groups <= 0 || g <= 1 || phase.m <= 0 || phase.n <= 0) {
        return -1;
    }
    if (phase.kind == GqaBandPhaseKind::OVERLAP_LOWER || phase.kind == GqaBandPhaseKind::OVERLAP_UPPER) {
        return GqaBandOverlapChunkKvRunUpperBound(k, groups, g, phase.m, phase.n);
    }
    if (phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE || phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE) {
        const int64_t columnWaves = phase.n / k;
        // Full groups, the odd group and the ell remainder keep each column
        // contiguous across heads. In paired waves t>=2, cores 2..k start
        // inside a column and revisit it after the paired column (A/B/A).
        const int64_t revisits = columnWaves >= 2 ? (groups % k) / 2 * columnWaves * (k - 1) : 0;
        return groups * phase.n + revisits;
    }
    if (IsGqaBandEdgePair(phase)) {
        if (CanReorderGqaBandEdgePair(phase, k, groups, g)) {
            return 2 * groups * phase.n;
        }
        // Without the proven head-inner permutation, a cyclic row interval
        // can visit its two physical columns as A/B/A for every head ID.
        return 3 * groups * phase.n * g;
    }
    if (phase.kind == GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR) {
        return groups * GqaBandDenseKvRunUpperBound(k, phase.m, phase.n, 1, g, false);
    }
    if (phase.kind == GqaBandPhaseKind::DIAGONAL_COLUMN_OWNER) {
        return groups * phase.n;
    }
    if (phase.kind == GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER ||
        phase.kind == GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER ||
        phase.kind == GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER) {
        const GqaBandDomain domain = DecodeGqaBandFoldedDomain(phase);
        return CanUseGqaBandFoldedColumnOwner(k, groups, g, domain.m, domain.n, domain.p, domain.q) ?
                   groups * GqaBandMin(domain.n, domain.m + domain.q - 1) :
                   -1;
    }
    if (phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER) {
        return phase.m == phase.n && CanUseGqaBandColumnOwner(k, groups, g, phase.m, phase.n, phase.m, 1) ?
                   groups * phase.n :
                   -1;
    }
    if (phase.kind == GqaBandPhaseKind::BAND_COLUMN_OWNER ||
        phase.kind == GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER) {
        return CanUseGqaBandColumnOwner(k, groups, g, phase.m, phase.n, phase.rowOffset, phase.colOffset) ?
                   groups * (phase.n < phase.m + phase.colOffset - 1 ? phase.n : phase.m + phase.colOffset - 1) :
                   -1;
    }
    return GqaBandDenseKvRunUpperBound(k, phase.m, phase.n, groups, g, phase.kind == GqaBandPhaseKind::DIAGONAL);
}

inline int64_t GqaBandChunkKvRunUpperBound(const GqaBandPlan& plan, bool tail)
{
    if (plan.k <= 0 || plan.groupsPerChunk <= 0 || plan.b <= 0) {
        return -1;
    }
    const int64_t groups = tail ? plan.b % plan.groupsPerChunk : plan.groupsPerChunk;
    if (groups == 0) {
        return 0;
    }
    const bool independentTail = tail && plan.tailPhaseCount > 0;
    const int64_t phaseCount = independentTail ? plan.tailPhaseCount : plan.phaseCount;
    int64_t runs = 0;
    for (int64_t i = 0; i < phaseCount; ++i) {
        const GqaBandPhase& phase = independentTail ? plan.tailPhases[i] : plan.phases[i];
        const int64_t phaseRuns = GqaBandPhaseKvRunUpperBound(phase, plan.k, groups, plan.g);
        if (phaseRuns < 0) {
            return -1;
        }
        runs += phaseRuns;
    }
    return runs;
}

inline int64_t GqaBandKvRunUpperBound(const GqaBandPlan& plan)
{
    const int64_t fullRuns = GqaBandChunkKvRunUpperBound(plan, false);
    const int64_t tailRuns = GqaBandChunkKvRunUpperBound(plan, true);
    if (fullRuns < 0 || tailRuns < 0) {
        return -1;
    }
    // Adjacent phases may merge their last/first run. Summing without that
    // merge is an upper bound, including independent/shared tail geometries.
    return plan.b / plan.groupsPerChunk * fullRuns + tailRuns;
}

// Return an exact cost only for existing schedules whose per-phase run count
// and lack of cross-phase merging are proved. Other schedules retain the safer
// physical-column lower bound at the call site; two upper bounds cannot prove
// that a new schedule preserves the currently selected K/V reuse.
inline int64_t GqaBandScheduleKvRunExact(const GqaBandSchedule& schedule)
{
    if (schedule.k <= 0 || schedule.b <= 0 || schedule.g <= 1 || schedule.groupsPerChunk <= 0 ||
        schedule.phaseCount <= 0 || schedule.phaseCount > GQA_BAND_MAX_PHASES || schedule.tailPhaseCount < 0 ||
        schedule.tailPhaseCount > GQA_BAND_MAX_PHASES) {
        return -1;
    }
    int64_t totalRuns = 0;
    for (int64_t tail = 0; tail < 2; ++tail) {
        const int64_t groups = tail != 0 ? schedule.b % schedule.groupsPerChunk : schedule.groupsPerChunk;
        if (groups == 0) {
            continue;
        }
        const bool separateTail = tail != 0 && schedule.tailPhaseCount > 0;
        const int64_t count = separateTail ? schedule.tailPhaseCount : schedule.phaseCount;
        const GqaBandPhase* phases = separateTail ? schedule.tailPhases : schedule.phases;
        int64_t chunkRuns = 0;
        for (int64_t i = 0; i < count; ++i) {
            const GqaBandPhase& phase = phases[i];
            if (phase.m <= 0 || phase.n <= 0) {
                return -1;
            }
            if (count == 1 && (phase.kind == GqaBandPhaseKind::BAND_COLUMN_OWNER ||
                               phase.kind == GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER ||
                               phase.kind == GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER ||
                               phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER)) {
                const int64_t runs = GqaBandPhaseKvRunUpperBound(phase, schedule.k, groups, schedule.g);
                if (runs < 0) {
                    return -1;
                }
                chunkRuns += runs;
                continue;
            }
            if (phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE || phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE) {
                if (phase.m != phase.n || phase.n / schedule.k >= 2) {
                    return -1;
                }
                chunkRuns += groups * phase.n;
            } else if (phase.kind == GqaBandPhaseKind::RECTANGLE ||
                       phase.kind == GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR) {
                const int64_t denseGroups = phase.kind == GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR ? 1 : groups;
                const int64_t denseRound =
                    GqaBandDenseRounds(schedule.k, phase.m, phase.n, denseGroups, schedule.g) / phase.m;
                if (denseRound / GqaBandGcd(denseGroups * schedule.g, denseRound) * phase.m < phase.n) {
                    return -1;
                }
                const int64_t copies = phase.kind == GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR ? groups : 1;
                chunkRuns +=
                    copies * GqaBandDenseKvRunUpperBound(schedule.k, phase.m, phase.n, denseGroups, schedule.g, false);
            } else {
                return -1;
            }
            for (int64_t j = 0; j < i; ++j) {
                if (phase.colOffset < phases[j].colOffset + phases[j].n &&
                    phases[j].colOffset < phase.colOffset + phase.n) {
                    return -1;
                }
            }
        }
        totalRuns += chunkRuns * (tail != 0 ? 1 : schedule.b / schedule.groupsPerChunk);
    }
    return totalRuns;
}

inline int64_t GqaBandLegacyKvRunLowerBound(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g)
{
    const int64_t validColumns = GqaBandMin(n, m + q - 1);
    k = GqaBandMin(k, GqaBandMin(b * g * m, b * n));
    const bool symmetricPair = 2 * b == k && m - p == validColumns - q && m > 1 && m - p > 0;
    if (b < k && p + q > m && p + q > validColumns && !symmetricPair) {
        const int64_t denseRound = GqaBandDenseRounds(k, m, validColumns, b, g) / m;
        if (denseRound / GqaBandGcd(b * g, denseRound) * m >= validColumns) {
            // This legacy branch is ordinary Dense followed by the BAND mask.
            // Every physical head-column is nonempty; filtering rows preserves
            // its exact run count when there is no Dense ID correction.
            return GqaBandDenseKvRunUpperBound(k, m, validColumns, b, g, false);
        }
    }
    // Never call the ideal number of physical columns the legacy exact cost.
    // Folded/symmetric legacy branches can have many more runs than this.
    return b * validColumns;
}

inline bool HasGqaBandQueryLocality(const GqaBandPlan& plan)
{
    if (plan.k <= 0 || plan.groupsPerChunk <= 0 || plan.g <= 1) {
        return false;
    }
    for (int64_t tail = 0; tail < 2; ++tail) {
        const int64_t groups = tail != 0 ? plan.b % plan.groupsPerChunk : plan.groupsPerChunk;
        if (groups == 0) {
            continue;
        }
        const bool independentTail = tail != 0 && plan.tailPhaseCount > 0;
        const int64_t phaseCount = independentTail ? plan.tailPhaseCount : plan.phaseCount;
        for (int64_t i = 0; i < phaseCount; ++i) {
            const GqaBandPhase& phase = independentTail ? plan.tailPhases[i] : plan.phases[i];
            if (phase.kind != GqaBandPhaseKind::DIAGONAL || phase.m <= 1 || phase.m >= plan.k || phase.n <= 1) {
                continue;
            }
            // Thin strips use NoTune, unlike the legacy folded Dense mapping.
            // For adjacent columns of the same Q row, ID advances by H*G and
            // local row decreases by one. Only a zero residue guarantees a
            // one-round gap instead of a long wrap through the head-ID range.
            const int64_t denseRound = GqaBandDenseRounds(plan.k, phase.m, phase.n, groups, plan.g) / phase.m;
            if ((groups * plan.g) % denseRound != 0) {
                return false;
            }
        }
    }
    // This guards the thin-strip regression; it is not a cache-hit guarantee
    // for other geometries or for transitions between different phases.
    return true;
}

// Geometry order is part of stable tie selection for both host search stages.
constexpr GqaBandGeometryMode GQA_BAND_GEOMETRY_MODES[] = {GqaBandGeometryMode::UPPER_TRIANGLE,
                                                           GqaBandGeometryMode::GENERAL,
                                                           GqaBandGeometryMode::REVERSED_UPPER_TRIANGLE,
                                                           GqaBandGeometryMode::REVERSED_GENERAL,
                                                           GqaBandGeometryMode::PAIRED_EDGES,
                                                           GqaBandGeometryMode::GENERAL_PAIRED_EDGES};
constexpr int64_t GQA_BAND_GEOMETRY_COUNT = sizeof(GQA_BAND_GEOMETRY_MODES) / sizeof(GQA_BAND_GEOMETRY_MODES[0]);

// Enum values describe geometry; their position here describes stable tie order.
constexpr int64_t GqaBandGeometryIndex(GqaBandGeometryMode geometry)
{
    for (int64_t index = 0; index < GQA_BAND_GEOMETRY_COUNT; ++index) {
        if (GQA_BAND_GEOMETRY_MODES[index] == geometry) {
            return index;
        }
    }
    return GQA_BAND_GEOMETRY_COUNT; // Only the declared geometry modes are valid cache keys.
}

struct GqaBandGeometryMetrics {
    int64_t rounds = -1;
    int64_t kvRuns = -1;
    bool queryLocality = false;
};

// Host-only. Cache six geometry skeletons and one small summary per group size,
// not a full/tail plan per candidate. A single allocation is bounded by K.
class GqaBandCandidateCache {
public:
    GqaBandCandidateCache(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g,
                          int64_t requestedGroups = 0)
        : k_(k),
          b_(b),
          g_(g),
          maxGroups_(GqaBandMin(b, k)),
          requestedGroups_(requestedGroups),
          metrics_((maxGroups_ +
                    (requestedGroups > maxGroups_ && requestedGroups != b ? REQUESTED_TAIL_SLOT_OFFSET :
                                                                            UNSPLIT_GROUP_SLOT_OFFSET) +
                    1) *
                   GQA_BAND_GEOMETRY_COUNT)
    {
        for (int64_t mode = 0; mode < GQA_BAND_GEOMETRY_COUNT; ++mode) {
            GqaBandPlan geometry;
            if (!BuildGqaBandGeometryByMode(m, n, p, q, GQA_BAND_GEOMETRY_MODES[mode], geometry)) {
                continue;
            }
            geometries_[mode].phaseCount = geometry.phaseCount;
            geometries_[mode].mode = geometry.geometryMode;
            for (int64_t phase = 0; phase < geometry.phaseCount; ++phase) {
                geometries_[mode].phases[phase] = geometry.phases[phase];
            }
        }
    }

    bool HasGeometry(int64_t mode) const
    {
        return geometries_[mode].phaseCount > 0;
    }

    const GqaBandGeometryMetrics& GetMetrics(int64_t mode, int64_t groups)
    {
        if (groups == 0) {
            return emptyMetrics_;
        }
        // The shared host search uses 1..min(B,K) and B. The public one-group
        // builder also preserves arbitrary H>K and its possibly large tail.
        const int64_t slot =
            groups <= maxGroups_ ?
                groups :
                maxGroups_ + (groups == b_ ? UNSPLIT_GROUP_SLOT_OFFSET :
                                             (groups == requestedGroups_ ? REQUESTED_GROUP_SLOT_OFFSET :
                                                                           REQUESTED_TAIL_SLOT_OFFSET));
        auto& metrics = metrics_[slot * GQA_BAND_GEOMETRY_COUNT + mode];
        if (metrics.rounds < 0) {
            GqaBandPlan plan;
            LoadGeometry(mode, plan);
            SetGqaBandGroups(k_, groups, g_, groups, plan);
            ConfigureGqaBandDensePhases(plan);
            metrics.rounds = plan.totalRounds;
            metrics.kvRuns = GqaBandKvRunUpperBound(plan);
            metrics.queryLocality = HasGqaBandQueryLocality(plan);
        }
        return metrics;
    }

    void Build(int64_t groups, int64_t fullMode, int64_t tailMode, bool separateTail, GqaBandPlan& plan) const
    {
        LoadGeometry(fullMode, plan);
        if (!separateTail) {
            SetGqaBandGroups(k_, b_, g_, groups, plan);
            ConfigureGqaBandDensePhases(plan);
            return;
        }
        SetGqaBandGroups(k_, groups, g_, groups, plan);
        ConfigureGqaBandDensePhases(plan);
        GqaBandPlan tail;
        LoadGeometry(tailMode, tail);
        SetGqaBandGroups(k_, b_ % groups, g_, b_ % groups, tail);
        ConfigureGqaBandDensePhases(tail);
        plan.b = b_;
        plan.tailGeometryMode = tail.geometryMode;
        plan.tailPhaseCount = tail.phaseCount;
        plan.tailRounds = tail.totalRounds;
        plan.totalRounds = b_ / groups * plan.fullRounds + plan.tailRounds;
        for (int64_t phase = 0; phase < tail.phaseCount; ++phase) {
            plan.tailPhases[phase] = tail.phases[phase];
            plan.tailPhases[phase].tailRounds = tail.phases[phase].fullRounds;
        }
    }

private:
    // Slots 1..maxGroups_ are indexed by group count. Larger requests must
    // not alias the unsplit group, the requested full group, or its tail.
    static constexpr int64_t UNSPLIT_GROUP_SLOT_OFFSET = 1;
    static constexpr int64_t REQUESTED_GROUP_SLOT_OFFSET = 2;
    static constexpr int64_t REQUESTED_TAIL_SLOT_OFFSET = 3;

    struct Geometry {
        GqaBandPhase phases[GQA_BAND_MAX_PHASES];
        int64_t phaseCount = 0;
        GqaBandGeometryMode mode = GqaBandGeometryMode::GENERAL;
    };

    void LoadGeometry(int64_t mode, GqaBandPlan& plan) const
    {
        plan = GqaBandPlan{};
        plan.geometryMode = geometries_[mode].mode;
        plan.phaseCount = geometries_[mode].phaseCount;
        for (int64_t phase = 0; phase < plan.phaseCount; ++phase) {
            plan.phases[phase] = geometries_[mode].phases[phase];
        }
    }

    int64_t k_;
    int64_t b_;
    int64_t g_;
    int64_t maxGroups_;
    int64_t requestedGroups_;
    Geometry geometries_[GQA_BAND_GEOMETRY_COUNT];
    std::vector<GqaBandGeometryMetrics> metrics_;
    const GqaBandGeometryMetrics emptyMetrics_{0, 0, true};
};

// Original mapper cost used for candidate admission; not a final fallback-plan writer.
inline int64_t GqaBandLegacyKvRunExact(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g)
{
    k = GqaBandMin(k, GqaBandMin(b * g * m, b * n));
    if (k <= 0 || m <= 0 || n <= 0 || p <= 0 || q <= 0 || b <= 0 || g <= 1) {
        return -1;
    }
    int64_t rowExtent = m;
    if (p + q > m) {
        n = q + GqaBandMin(m - 1, n - q);
    } else {
        const int64_t middle = GqaBandMin(n - q + 1, m + 2 - p - q);
        const int64_t last = GqaBandMax(0, GqaBandMin(p + n - m - 1, p + q - 2));
        if (last == 0) {
            rowExtent = p + q + middle - 2;
        }
        n = q - 1 + middle + last;
    }
    // CalPosWholeBatch visits each column once per head. Consecutive heads
    // merge only when this group has exactly one physical column.
    const int64_t completeRuns = (b / k) * k * n * (n == 1 ? 1 : g);
    const int64_t tail = b % k;
    if (tail == 0) {
        return completeRuns;
    }
    int64_t denseRows;
    bool noTune = false;
    if (p + q > m) {
        const bool symmetricPair = 2 * tail == k && m - p == n - q && m > 1 && m - p > 0;
        if (p + q <= n || symmetricPair) {
            return -1;
        }
        denseRows = m;
    } else {
        if (p + q > rowExtent) {
            return -1;
        }
        denseRows = p + q - 1;
        if (n > m) {
            const int64_t rowFoldRounds = GqaBandDenseRounds(k, rowExtent, denseRows, tail, g);
            const int64_t columnRounds = GqaBandDenseRounds(k, denseRows, n, tail, g);
            if (rowFoldRounds <= columnRounds) {
                // This branch folds virtual columns into different physical columns.
                return -1;
            }
            noTune = true;
        }
    }
    const int64_t denseRound = GqaBandDenseRounds(k, denseRows, n, tail, g) / denseRows;
    if (!noTune && denseRound / GqaBandGcd(tail * g, denseRound) * denseRows < n) {
        return -1;
    }
    // Every head-column is nonempty in the complete physical tile domain.
    // Row filtering/folding leaves its column and contiguous head-ID run intact.
    // Complete groups and the tail address disjoint KV groups and cannot merge.
    return completeRuns + GqaBandDenseKvRunUpperBound(k, denseRows, n, tail, g, noTune);
}

// MHA BAND domain normalization and source-compatible round bound.
inline DeterBandScheduleParams NormalizeDeterBandScheduleParams(int64_t m, int64_t n, int64_t p, int64_t q)
{
    // Keep the clamp and shifted-BAND conversion identical to CalDeterMaxLoopNum on the kernel side.
    p = std::min(p, m);
    q = std::min(q, n);
    if (p < 0) {
        return {m, n + p, 1, p + q};
    }
    if (q < 0) {
        return {m + q, n, p + q, 1};
    }
    return {m, n, p, q};
}

inline int64_t CalcLegacyDeterBandMaxRound(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                           int64_t hostMaxRound)
{
    // Keep rm2 identical to GenBandInfo and CalDeterMaxLoopNum on the kernel side.
    const int64_t rm2 =
        p + q > m ? m * PositiveCeilDiv(n * b, std::min(k, b * m)) : PositiveCeilDiv(n * b, k) * (p + q - 1);
    return std::max(hostMaxRound, rm2);
}
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_BAND_CANDIDATES_H_
