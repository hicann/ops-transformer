/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_BAND_SELECTOR_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_BAND_SELECTOR_H_

// MHA/GQA BAND selection policy, ordered search and candidate admission.
#include <algorithm>
#include <climits>
#include <limits>
#include "candidates.h"
#include "../candidate_types.h"
#include "../causal.h"
#include "../fag_deter_utils.h"

namespace optiling {
namespace fag_pto_arch35 {
constexpr int64_t DETER_BAND_SWIZZLE_PERCENT_BASE = 100, DETER_BAND_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT = 90,
                  DETER_BAND_SWIZZLE_MAX_ROUND_GROWTH_PERCENT = 3;

inline bool IsRoundGrowthWithinLimit(int64_t candidateRound, int64_t legacyRound, int64_t maxGrowthPercent,
                                     int64_t percentBase)
{
    return candidateRound > 0 && legacyRound > 0 &&
           candidateRound * percentBase <= legacyRound * (percentBase + maxGrowthPercent);
}

// MHA BAND candidates share the original Dense/Causal embedding and admission.
inline DeterBandScheduleResult SelectDeterBandSchedule(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                                       int64_t hostMaxRound)
{
    DeterBandScheduleResult result;
    if (k <= 0 || m <= 0 || n <= 0 || p <= 0 || q <= 0 || b <= 0) {
        return result;
    }

    // Keep this preprocessing identical to GenBandHybridInfo on the kernel side.
    p = std::min(p, m);
    q = std::min(q, n);
    const int64_t legacyMaxRound = CalcLegacyDeterBandMaxRound(k, m, n, p, q, b, hostMaxRound);
    m = std::min(m, n + p - 1);
    n = std::min(n, m + q - 1);

    int64_t l1 = 0;
    int64_t l2 = 0;
    int64_t l3 = 0;
    int64_t bandBlocks = 0;
    if (p + q <= m) {
        l1 = q - 1;
        l2 = std::min(n - q + 1, m + NUM_TWO - p - q);
        l3 = std::max<int64_t>(0, std::min(p + n - m - 1, p + q - NUM_TWO));
        bandBlocks = (NUM_TWO * p - NUM_TWO + q) * l1 / NUM_TWO + (p + q - 1) * l2 + (p + q - NUM_TWO) * l3 -
                     l3 * (l3 - 1) / NUM_TWO;
    } else {
        l1 = m - p;
        l2 = p + q - m;
        l3 = std::min(n - q, m - 1);
        bandBlocks = (p + m - 1) * l1 / NUM_TWO + m * l2 + (NUM_TWO * m - 1 - l3) * l3 / NUM_TWO;
    }

    const int64_t pairCount = std::max<int64_t>(0, std::min(l1, l3 - p + 1));
    const int64_t colsPerBatch = l1 + l2 + l3 - pairCount;
    const int64_t bandSlot = p + q - 1;
    result.mode = DeterBandScheduleMode::BAND;
    result.maxRound = PositiveCeilDiv(b * colsPerBatch, k) * bandSlot;
    result.m = m;
    result.n = n;
    result.p = p;
    result.q = q;

    // Dense is a safe superset only when all k cores remain active and the columns issued to one
    // batch in a round cannot wrap by m and hit the same row. This is less restrictive than m >= k.
    const int64_t denseK = std::min(k, m * b);
    if (denseK == k && std::min(denseK, n) <= m) {
        const int64_t denseRound = PositiveCeilDiv(n * b, denseK) * m;
        if (denseRound < result.maxRound) {
            result.mode = DeterBandScheduleMode::DENSE;
            result.maxRound = denseRound;
        }
    }

    // Embed BAND into the smaller enclosing causal triangle. Upper embedding rotates both axes
    // instead of transposing them, preserving fixed-core ownership of each original KV column.
    // IsValidForDeter removes the padding and the opposite window boundary in either orientation.
    const int64_t lowerCausalSize = m + q - 1;
    const int64_t upperCausalSize = n + p - 1;
    const int64_t lowerWaste = lowerCausalSize * (lowerCausalSize + 1) / NUM_TWO - bandBlocks;
    const int64_t upperWaste = upperCausalSize * (upperCausalSize + 1) / NUM_TWO - bandBlocks;
    const bool useUpperCausal = upperWaste < lowerWaste;
    const int64_t causalSize = useUpperCausal ? upperCausalSize : lowerCausalSize;
    const int64_t causalWaste = useUpperCausal ? upperWaste : lowerWaste;
    constexpr int64_t causalPaddingBudgetDivisor = 10;
    const bool useCausal =
        bandBlocks > 0 && causalWaste >= 0 && causalWaste <= (bandBlocks - 1) / causalPaddingBudgetDivisor;
    if (useCausal) {
        const int64_t causalPairCount = b / NUM_TWO;
        const int64_t causalK = std::min(k, causalSize * causalPairCount);
        // CalCausalSwizzleIndex uses the dense swizzle internally, so do not select it when that
        // helper would reduce the active-core count below k.
        if (causalPairCount > 0 && causalK == k && causalK <= causalSize) {
            const int64_t pairRound = PositiveCeilDiv((causalSize + 1) * causalPairCount, causalK) * causalSize;
            const int64_t tailRound = (b & 1) == 0 ? 0 : CalcCausalSingleBatchRound(k, causalSize);
            const int64_t causalRound = pairRound + tailRound;
            if (causalRound < result.maxRound) {
                result.mode = useUpperCausal ? DeterBandScheduleMode::CAUSAL_UPPER : DeterBandScheduleMode::CAUSAL;
                result.maxRound = causalRound;
            }
        }
    }

    // All hybrid schedules iterate k slots in each round. Reject low-utilization schedules because
    // invalid coordinates still incur round traversal and index-calculation overhead in the kernel.
    const int64_t validSlotNum = bandBlocks * b;
    const int64_t totalSlotNum = k * result.maxRound;
    const bool slotUtilizationEnough =
        totalSlotNum > 0 && validSlotNum * DETER_BAND_SWIZZLE_PERCENT_BASE >=
                                totalSlotNum * DETER_BAND_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT;
    const bool roundGrowthWithinLimit = IsRoundGrowthWithinLimit(
        result.maxRound, legacyMaxRound, DETER_BAND_SWIZZLE_MAX_ROUND_GROWTH_PERCENT, DETER_BAND_SWIZZLE_PERCENT_BASE);
    if (!slotUtilizationEnough || !roundGrowthWithinLimit) {
        return {};
    }
    return result;
}

inline void CalcleBandDeterParam(FixedCandidateState& fBaseParams)
{
    int64_t m{fBaseParams.s1Outer}, n{fBaseParams.s2Outer}, k{static_cast<int64_t>(fBaseParams.aicNum)},
        b{fBaseParams.b * fBaseParams.n2};
    int64_t actualCalcS1Token{fBaseParams.s1Token}, actualCalcS2Token{fBaseParams.s2Token};
    int64_t p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
    int64_t q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;

    q = q > n ? n : q;
    p = p > m ? m : p;

    // 负数场景变换
    if (q < 0) {
        m = m + q;
        p = p + q;
        q = 1;
    } else if (p < 0) {
        n = n + p;
        q = p + q;
        p = 1;
    }

    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t L1, L2, L3, n_seg;
    if (p + q > m) {
        L1 = m - p;
        L2 = p + q - m;
        L3 = std::min(m - 1, n - q);
        n_seg = L1 + L2 + L3;
    } else {
        L1 = q - 1;
        L2 = std::min(n - q + 1, m + NUM_TWO - p - q);
        L3 = std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - NUM_TWO));
        if (L3 == 0) {
            m = p + q + L2 - NUM_TWO;
        }
        n_seg = L1 + L2 + L3;
    }
    int64_t r1 = (m * n_seg - (m - p) * (m - p + 1) / NUM_TWO - (n_seg - q) * (n_seg - q + 1) / NUM_TWO) * b1;
    int64_t r2 = 0;
    if (b2 > 0) {
        if (p + q > m) {
            r2 = std::max(m * CeilDivideBy((n * b2), std::min(k, b2 * m)), n);
        } else {
            r2 = std::max(CeilDivideBy((n * b2), k) * (p + q - 1), n);
        }
    }
    fBaseParams.deterMaxRound = r1 + r2;
}

// Preserve the original geometry order, strict comparisons and full/tail tie behavior.
inline void BuildLegacyGqaBandCandidate(GqaBandCandidateCache& cache, int64_t b, int64_t groups, GqaBandPlan& plan)
{
    int64_t uniformMode = GqaBandGeometryIndex(GqaBandGeometryMode::GENERAL);
    int64_t fullMode = uniformMode, tailMode = uniformMode;
    int64_t uniformRounds = 0, fullRounds = 0, tailRounds = 0;
    bool first = true;
    const int64_t fullChunks = b / groups;
    for (int64_t mode = 0; mode < GQA_BAND_GEOMETRY_COUNT; ++mode) {
        if (!cache.HasGeometry(mode)) {
            continue;
        }
        const int64_t full = cache.GetMetrics(mode, groups).rounds;
        const int64_t tail = cache.GetMetrics(mode, b % groups).rounds;
        const int64_t total = fullChunks * full + tail;
        if (GQA_BAND_GEOMETRY_MODES[mode] == GqaBandGeometryMode::GENERAL_PAIRED_EDGES) {
            const int64_t previous = GqaBandMin(uniformRounds, fullChunks * fullRounds + tailRounds);
            const int64_t combined = fullChunks * GqaBandMin(fullRounds, full) + GqaBandMin(tailRounds, tail);
            if (GqaBandMin(total, combined) >= previous) {
                continue;
            }
        }
        if (first || total < uniformRounds) {
            uniformMode = mode;
            uniformRounds = total;
        }
        if (first || full < fullRounds) {
            fullMode = mode;
            fullRounds = full;
        }
        if (first || tail < tailRounds) {
            tailMode = mode;
            tailRounds = tail;
        }
        first = false;
    }
    const bool separate = b % groups != 0 && fullChunks * fullRounds + tailRounds < uniformRounds;
    cache.Build(groups, separate ? fullMode : uniformMode, tailMode, separate, plan);
}

inline void BuildGqaBandCandidate(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g,
                                  int64_t groups, GqaBandPlan& plan)
{
    GqaBandCandidateCache cache(k, m, n, p, q, b, g, groups);
    BuildLegacyGqaBandCandidate(cache, b, groups, plan);
}

inline bool PreferGqaBandLocalityOnTie(const GqaBandPlan& plan)
{
    if (plan.k <= 0 || plan.b <= 1 || plan.tailPhaseCount != 0) {
        return false;
    }
    if (plan.groupsPerChunk == 1 && plan.phaseCount == 1) {
        const GqaBandPhase& phase = plan.phases[0];
        // For n=s*k, m>=k and s>1, both mappings keep whole KV columns on
        // each core. Serializing groups reduces their completion spans.
        return phase.kind == GqaBandPhaseKind::DIAGONAL && phase.rowOffset == 0 && phase.colOffset == 0 &&
               phase.m >= plan.k && phase.n > plan.k && phase.n % plan.k == 0;
    }
    // With b=t*k+1 and even k, H=2 consumes the complete groups in pairs,
    // then one tail group. For a k-by-k rectangle plus a k-by-k triangle,
    // rounds equal t*k*(3*k+1)*g/2 + 2*k*g, exactly the legacy bound.
    // Each pair stays within two KV groups; the caller still requires the
    // existing KV-run and query-locality checks before admitting this tie.
    constexpr int64_t pairedGroups = 2;
    constexpr int64_t rectanglePhaseIndex = 0;
    constexpr int64_t trianglePhaseIndex = 1;
    constexpr int64_t rectangleTrianglePhaseCount = 2;
    if (plan.groupsPerChunk != pairedGroups || plan.g <= 1 || plan.k % pairedGroups != 0 || plan.b <= plan.k ||
        plan.b % plan.k != 1 || plan.phaseCount != rectangleTrianglePhaseCount) {
        return false;
    }
    const GqaBandPhase& rectangle = plan.phases[rectanglePhaseIndex];
    const GqaBandPhase& triangle = plan.phases[trianglePhaseIndex];
    return rectangle.kind == GqaBandPhaseKind::RECTANGLE && rectangle.m == plan.k && rectangle.n == plan.k &&
           rectangle.rowOffset == 0 && rectangle.colOffset == 0 && triangle.kind == GqaBandPhaseKind::LOWER_TRIANGLE &&
           triangle.m == plan.k && triangle.n == plan.k && triangle.rowOffset == 0 && triangle.colOffset == plan.k;
}

// Shared host iteration only. Preserve each caller's candidate order; a
// repeated unsplit group cannot replace the same plan on a strict tie.
class GqaBandGroupRange {
public:
    class Iterator {
    public:
        Iterator(int64_t index, int64_t limit, int64_t all)
            : index_(index),
              limit_(limit),
              all_(all)
        {}
        int64_t operator*() const
        {
            return index_ <= limit_ ? index_ : all_;
        }
        Iterator& operator++()
        {
            ++index_;
            return *this;
        }
        bool operator!=(const Iterator& other) const
        {
            return index_ != other.index_;
        }

    private:
        int64_t index_;
        int64_t limit_;
        int64_t all_;
    };

    GqaBandGroupRange(int64_t all, int64_t capacity, bool includeUnsplit)
        : all_(all),
          limit_(std::min(all, capacity)),
          count_(limit_ + (includeUnsplit && all > limit_ ? 1 : 0))
    {}
    Iterator begin() const
    {
        return Iterator(1, limit_, all_);
    }
    Iterator end() const
    {
        return Iterator(count_ + 1, limit_, all_);
    }

private:
    int64_t all_;
    int64_t limit_;
    int64_t count_;
};

struct GqaBandAdmission {
    bool queryLocality = false;
    bool completeChunks = false;
    bool originalLocalityTie = false;
    bool allowEqualRounds = false;
};

inline bool TrySaveGqaBandCandidate(FixedCandidateState& params, const GqaBandPlan& plan, int64_t kvLimit,
                                    int64_t& bestRounds, int64_t& bestGroups, GqaBandAdmission admission = {})
{
    const bool equalAllowed = admission.allowEqualRounds ||
                              (admission.originalLocalityTie && bestGroups == 0 && PreferGqaBandLocalityOnTie(plan));
    if (plan.totalRounds <= 0 || plan.totalRounds > bestRounds || (plan.totalRounds == bestRounds && !equalAllowed) ||
        plan.totalRounds >= std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (admission.completeChunks &&
        (plan.fullRounds <= 0 || (plan.b % plan.groupsPerChunk != 0 && plan.tailRounds <= 0))) {
        return false;
    }
    const int64_t kvRuns = GqaBandKvRunUpperBound(plan);
    if (kvRuns < 0 || kvRuns > kvLimit || (admission.queryLocality && !HasGqaBandQueryLocality(plan))) {
        return false;
    }
    bestRounds = plan.totalRounds;
    bestGroups = plan.groupsPerChunk;
    SaveGqaBandSchedule(plan, params.deterGqaBandSchedule);
    return true;
}

// Search the admissible full/tail frontier instead of first discarding every
// geometry except the minimum-round one. The six modes and their pairs bound
// host work independently of sequence length; no task table is serialized.
inline void SelectGqaBandJointSchedule(FixedCandidateState& params, int64_t k, int64_t m, int64_t n, int64_t q,
                                       int64_t b, int64_t validColumns, int64_t legacyKvLimit,
                                       GqaBandCandidateCache& cache, int64_t& bestRounds, int64_t& bestGroups)
{
    int64_t referenceKvLimit = legacyKvLimit;
    if (bestGroups != 0) {
        const int64_t cubeBase = static_cast<int64_t>(ConstAxisTemplateNum::NUM128);
        const int64_t exact =
            params.s1 % cubeBase == 0 && params.s2 % cubeBase == 0 && validColumns == GqaBandMin(n, m + q - 1) ?
                GqaBandScheduleKvRunExact(params.deterGqaBandSchedule) :
                -1;
        referenceKvLimit = GqaBandMin(referenceKvLimit, exact >= 0 ? exact : b * validColumns);
    }
    // Freeze the complete previous selector's cost at entry. Candidate
    // improvements never change this search's admission baseline.
    for (const int64_t groups : GqaBandGroupRange(b, k, true)) {
        const int64_t fullChunks = b / groups;
        const int64_t tailGroups = b % groups;
        for (int64_t fullMode = 0; fullMode < GQA_BAND_GEOMETRY_COUNT; ++fullMode) {
            if (!cache.HasGeometry(fullMode)) {
                continue;
            }
            const auto& full = cache.GetMetrics(fullMode, groups);
            if (full.rounds <= 0 || full.kvRuns < 0 || !full.queryLocality) {
                continue;
            }
            for (int64_t tailMode = 0; tailMode < (tailGroups != 0 ? GQA_BAND_GEOMETRY_COUNT : 1); ++tailMode) {
                if (tailGroups != 0 && !cache.HasGeometry(tailMode)) {
                    continue;
                }
                const auto& tail = cache.GetMetrics(tailMode, tailGroups);
                if (tailGroups != 0 && (tail.rounds <= 0 || tail.kvRuns < 0 || !tail.queryLocality)) {
                    continue;
                }
                const int64_t totalRounds = fullChunks * full.rounds + tail.rounds;
                const int64_t kvRuns = fullChunks * full.kvRuns + tail.kvRuns;
                if (totalRounds <= 0 || totalRounds >= bestRounds ||
                    totalRounds >= std::numeric_limits<uint32_t>::max() || kvRuns > referenceKvLimit) {
                    continue;
                }
                GqaBandPlan candidate;
                cache.Build(groups, fullMode, tailMode, tailGroups != 0, candidate);
                TrySaveGqaBandCandidate(params, candidate, referenceKvLimit, bestRounds, bestGroups);
            }
        }
    }
}

// A common row displacement lets one head run on multiple disjoint row
// intervals while each physical KV column remains on exactly one core.
// This applies to the complete band geometry, including small head groups.
inline void SelectGqaBandGenericFoldedOwner(FixedCandidateState& params, int64_t k, int64_t m, int64_t n, int64_t p,
                                            int64_t q, int64_t b, int64_t validColumns, int64_t legacyKvLimit,
                                            int64_t& bestRounds, int64_t& bestGroups)
{
    const int64_t cubeBase = static_cast<int64_t>(ConstAxisTemplateNum::NUM128);
    const bool completePhysicalTiles = params.s1 % cubeBase == 0 && params.s2 % cubeBase == 0;
    // IsValidForDeter filters the rounded geometry with the original token
    // mask, including partial physical tiles. Its executed task stream is a
    // subsequence of this conflict-free mapper, so the geometric KV-run upper
    // bound remains valid. The previous mapper's geometric exact cost need not
    // survive filtering; use the touched physical columns as its lower bound.
    int64_t referenceKvLimit = completePhysicalTiles ? legacyKvLimit : b * validColumns;
    if (bestGroups != 0) {
        const int64_t exact = completePhysicalTiles && validColumns == std::min(n, m + q - 1) ?
                                  GqaBandScheduleKvRunExact(params.deterGqaBandSchedule) :
                                  -1;
        referenceKvLimit = std::min(referenceKvLimit, exact >= 0 ? exact : b * validColumns);
    }
    for (const int64_t groups : GqaBandGroupRange(b, k, true)) {
        GqaBandPlan candidate;
        AppendGqaBandFoldedPhase(candidate, GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER, {m, n, p, q, 0, 0});
        SetGqaBandGroups(k, b, params.g, groups, candidate);
        GqaBandAdmission admission;
        admission.completeChunks = true;
        TrySaveGqaBandCandidate(params, candidate, referenceKvLimit, bestRounds, bestGroups, admission);
    }
}

// One-sided bands can fold physical columns by the row period. The common
// row displacement gives distinct dQ rows even when there are fewer heads
// than active cores; fixed virtual-column ownership also protects shared dK/dV.
inline void SelectGqaBandFoldedOwner(FixedCandidateState& params, int64_t k, int64_t m, int64_t n, int64_t p, int64_t q,
                                     int64_t b, int64_t rowOffset, int64_t colOffset, int64_t legacyKvRuns,
                                     int64_t& bestRounds, int64_t& bestGroups)
{
    if ((p != 1 && q != 1) || (p == 1 && rowOffset != 0) || (p != 1 && colOffset != 0)) {
        return;
    }
    const bool upper = p == 1;
    const auto kind = upper ? GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER : GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER;
    for (const int64_t candidateGroups : GqaBandGroupRange(b, k, true)) {
        GqaBandPlan plan;
        // The width occupies the unused translation field for this orientation.
        AppendGqaBandFoldedPhase(plan, kind, {m, n, p, q, rowOffset, colOffset});
        SetGqaBandGroups(k, b, params.g, candidateGroups, plan);
        GqaBandAdmission admission;
        admission.completeChunks = true;
        TrySaveGqaBandCandidate(params, plan, legacyKvRuns, bestRounds, bestGroups, admission);
    }
}

// Signed mathematical ceil. Device math::Ceil uses (a+b-1)/b and truncates
// toward zero for negative a, so reproduce that separately for legacy offsets.
inline int64_t GqaBandSignedCeil(int64_t value, int64_t unit)
{
    return value / unit + (value % unit > 0 ? 1 : 0);
}

struct GqaBandNormalizedInput {
    GqaBandDomain domain;
    GqaBandDomain legacyDomain;
    int64_t validColumns = 0;
    bool negativeWindow = false;
};

// Keep the physical origin separate from p/q. The legacy origin is retained
// because CalBandDeterIndex already adds it after the scheduled mapper.
inline bool NormalizeGqaBandInput(const FixedCandidateState& params, int64_t k, int64_t m, int64_t n, int64_t b,
                                  int64_t cubeBase, GqaBandNormalizedInput& input)
{
    const int64_t pre = std::min<int64_t>(params.s1Token, INT32_MAX);
    const int64_t next = std::min<int64_t>(params.s2Token, INT32_MAX);
    input.negativeWindow = pre < 0 || next < 0;
    if (!input.negativeWindow) {
        const int64_t p = std::min(m, CeilDivideBy(pre, cubeBase) + 1);
        const int64_t q = std::min(n, CeilDivideBy(next, cubeBase) + 1);
        input.domain = {m, n, p, q, 0, 0};
        input.legacyDomain = input.domain;
        input.validColumns = std::min(
            n, params.s1 / cubeBase + next / cubeBase + CeilDivideBy(params.s1 % cubeBase + next % cubeBase, cubeBase));
        return true;
    }
    // These exact negative-window inequalities have only been proved for
    // complete physical tiles. Keep empty windows and legacy core guards.
    if (params.s1 % cubeBase != 0 || params.s2 % cubeBase != 0 || (pre < 0 && (next < 0 || pre < -next)) ||
        (next < 0 && pre >= 0 && next < -pre)) {
        return false;
    }
    const int64_t exactP = std::min(m, GqaBandSignedCeil(pre, cubeBase) + 1);
    const int64_t exactQ = std::min(n, GqaBandSignedCeil(next, cubeBase) + 1);
    const int64_t rowOrigin = std::max<int64_t>(0, 1 - exactQ);
    const int64_t colOrigin = std::max<int64_t>(0, 1 - exactP);
    if (rowOrigin >= m || colOrigin >= n) {
        return false;
    }
    const int64_t normalizedM = m - rowOrigin;
    const int64_t normalizedN = n - colOrigin;
    const int64_t p = std::min(normalizedM, exactP - rowOrigin + colOrigin);
    const int64_t q = std::min(normalizedN, exactQ + rowOrigin - colOrigin);
    if (p <= 0 || q <= 0) {
        return false;
    }
    input.domain = {normalizedM, normalizedN, p, q, rowOrigin, colOrigin};
    GqaBandDomain& legacy = input.legacyDomain;
    legacy = {m, n, std::min(m, (pre + cubeBase - 1) / cubeBase + 1), std::min(n, (next + cubeBase - 1) / cubeBase + 1),
              0, 0};
    if (legacy.p < 0) {
        legacy.n += legacy.p;
        legacy.colOrigin = -legacy.p;
        legacy.q += legacy.p;
        legacy.p = 1;
    } else if (legacy.q < 0) {
        legacy.m += legacy.q;
        legacy.rowOrigin = -legacy.q;
        legacy.p += legacy.q;
        legacy.q = 1;
    }
    if (legacy.m <= 0 || legacy.n <= 0 || legacy.p <= 0 || legacy.q <= 0 ||
        k > std::min(b * params.g * legacy.m, b * legacy.n)) {
        return false;
    }
    input.validColumns = std::min(normalizedN, normalizedM + q - 1);
    return true;
}

inline GqaBandDomain GetGqaBandMappingDomain(const GqaBandNormalizedInput& input)
{
    GqaBandDomain domain = input.domain;
    domain.rowOrigin -= input.legacyDomain.rowOrigin;
    domain.colOrigin -= input.legacyDomain.colOrigin;
    return domain;
}

// Called only after SelectGqaBandSchedule's common template/overflow guards.
// Only completely masked leading tiles are removed. Partial boundary tiles
// remain and continue to use the original sparse token mask in the kernel.
inline void SelectGqaBandNegativeSchedule(FixedCandidateState& params, int64_t k, int64_t b,
                                          const GqaBandNormalizedInput& input)
{
    const GqaBandDomain domain = GetGqaBandMappingDomain(input);
    const GqaBandDomain& legacy = input.legacyDomain;
    const int64_t normalizedM = domain.m;
    const int64_t normalizedN = domain.n;
    const int64_t p = domain.p;
    const int64_t q = domain.q;
    const int64_t legacyRounds = GqaBandLegacyRounds(k, legacy.m, legacy.n, legacy.p, legacy.q, b, params.g);
    // Every touched physical KV column must be submitted at least once.
    const int64_t legacyKvLower = b * std::min(normalizedN, normalizedM + q - 1);
    int64_t bestRounds = legacyRounds;
    int64_t bestGroups = 0;
    for (const int64_t candidateGroups : GqaBandGroupRange(b, k, true)) {
        GqaBandPlan plan;
        // GENERAL only: ordinary phase offsets are unconditional translations.
        // EDGE_PAIR offsets select one half and BAND_COLUMN_OWNER stores p/q there;
        // neither is safe to shift as an ordinary phase.
        BuildGqaBandGeneralGeometry(normalizedM, normalizedN, p, q, plan);
        SetGqaBandGroups(k, b, params.g, candidateGroups, plan);
        ConfigureGqaBandDensePhases(plan);
        for (int64_t i = 0; i < plan.phaseCount; ++i) {
            plan.phases[i].rowOffset += domain.rowOrigin;
            plan.phases[i].colOffset += domain.colOrigin;
        }
        GqaBandAdmission admission;
        admission.queryLocality = true;
        TrySaveGqaBandCandidate(params, plan, legacyKvLower, bestRounds, bestGroups, admission);
    }
    const int64_t validRows = std::min(normalizedM, normalizedN + p - 1);
    const int64_t validColumns = std::min(normalizedN, normalizedM + q - 1);
    // A translated lower triangle is the p=m,q=1 instance of the generic
    // column-owner mapper. All sizes, group counts and head counts use the
    // same load calculation; no negative-window shape whitelist is needed.
    if (q == 1 && p == normalizedM && validRows == validColumns &&
        CanUseGqaBandColumnOwner(k, b, params.g, validRows, validColumns, validRows, 1)) {
        GqaBandPlan plan;
        AppendGqaBandPhase(plan, GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER, validRows, validColumns,
                           domain.rowOrigin, domain.colOrigin);
        SetGqaBandGroups(k, b, params.g, b, plan);
        if (plan.totalRounds < legacyRounds) {
            GqaBandAdmission admission;
            admission.allowEqualRounds = true;
            TrySaveGqaBandCandidate(params, plan, legacyKvLower, bestRounds, bestGroups, admission);
        }
    }
    SelectGqaBandFoldedOwner(params, k, normalizedM, normalizedN, p, q, b, domain.rowOrigin, domain.colOrigin,
                             legacyKvLower, bestRounds, bestGroups);
    if (bestGroups != 0) {
        params.deterGqaBandScheduleMode = DeterGqaBandScheduleMode::PIECEWISE;
        params.deterGqaBandGroups = static_cast<uint32_t>(bestGroups);
        params.deterMaxRound = bestRounds;
    }
}

inline void SelectGqaBandOverlapSchedule(FixedCandidateState& params, int64_t k, int64_t m, int64_t n, int64_t p,
                                         int64_t q, int64_t b, int64_t validColumns, int64_t legacyKvLower,
                                         int64_t& bestRounds, int64_t& bestGroups)
{
    const int64_t cubeBase = static_cast<int64_t>(ConstAxisTemplateNum::NUM128);
    if (m < 2) {
        return;
    }
    const bool completePhysicalTiles = params.s1 % cubeBase == 0 && params.s2 % cubeBase == 0;
    // Masking removes tasks without creating conflicts or increasing the
    // candidate's KV-run upper bound. Compare against a physical lower bound
    // whenever a partial tile prevents using the previous plan's exact cost.
    int64_t currentKvLower = completePhysicalTiles ? legacyKvLower : b * validColumns;
    if (bestGroups != 0) {
        // A fully masked rounded column invalidates the exact geometric cost.
        const int64_t geometricColumns = std::min(n, m + q - 1);
        const int64_t exactRuns = completePhysicalTiles && validColumns == geometricColumns ?
                                      GqaBandScheduleKvRunExact(params.deterGqaBandSchedule) :
                                      -1;
        currentKvLower = exactRuns >= 0 ? exactRuns : b * validColumns;
    }
    for (const int64_t groups : GqaBandGroupRange(b, k / (m / 2 + 1), false)) {
        GqaBandPlan candidate;
        if (!BuildGqaBandOverlapCandidate(k, m, n, p, q, b, params.g, groups, candidate)) {
            continue;
        }
        TrySaveGqaBandCandidate(params, candidate, std::min(legacyKvLower, currentKvLower), bestRounds, bestGroups);
    }
}

inline void SelectGqaBandSchedule(FixedCandidateState& params, bool originalSquareBase)
{
    params.deterGqaBandScheduleMode = DeterGqaBandScheduleMode::ORIGINAL;
    params.deterGqaBandGroups = 0;
    params.deterGqaBandSchedule = GqaBandSchedule{};
    const int64_t cubeBase = static_cast<int64_t>(ConstAxisTemplateNum::NUM128);
    const int64_t k = static_cast<int64_t>(params.aicNum);
    const int64_t m = params.s1Outer;
    const int64_t n = params.s2Outer;
    // Split-mode keeps its existing implementation; negative tokens use the exact host normalization below.
    if (!params.isSwizzle || !originalSquareBase || !params.isDeterministic || params.layoutType == INPUT_FORMAT_TND ||
        params.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_BAND) ||
        params.splitAxis != SplitAxisEnum::BN2GS1S2 || params.g <= 1 || k <= 0 || k > MAX_CORE_NUM ||
        params.coreNum != k * NUM_TWO || m <= 0 || n <= 0 || params.b <= 0 || params.n2 <= 0 ||
        params.s1Inner * params.s1CvRatio != cubeBase || params.s2Inner * params.s2CvRatio != cubeBase) {
        return;
    }
    // Bound every intermediate before multiplication (geometry, group tails and phase sums).
    int64_t arithmeticBudget = std::numeric_limits<int64_t>::max() / (NUM_TWO * GQA_BAND_MAX_PHASES);
    const int64_t factors[] = {params.b, params.n2, params.g, m, n};
    for (int64_t factor : factors) {
        if (factor > arithmeticBudget) {
            return;
        }
        arithmeticBudget /= factor;
    }
    const int64_t b = params.b * params.n2;
    if (b > std::numeric_limits<uint32_t>::max()) {
        return;
    }
    GqaBandNormalizedInput input;
    if (!NormalizeGqaBandInput(params, k, m, n, b, cubeBase, input)) {
        return;
    }
    if (input.negativeWindow) {
        SelectGqaBandNegativeSchedule(params, k, b, input);
        return;
    }
    // SaveToTilingData applies the same clamp before the device builds p/q.
    const int64_t p = input.domain.p;
    const int64_t q = input.domain.q;
    const int64_t legacyRounds = GqaBandLegacyRounds(k, m, n, p, q, b, params.g);
    // Rounding S1 and nextTokens separately may add an entirely masked last
    // column. Count real touched columns before using the geometric lower bound.
    const int64_t validColumns = input.validColumns;
    const int64_t legacyKvRuns = validColumns < std::min(n, m + q - 1) ?
                                     b * validColumns :
                                     GqaBandLegacyKvRunLowerBound(k, m, n, p, q, b, params.g);
    GqaBandPlan plan;
    GqaBandCandidateCache cache(k, m, n, p, q, b, params.g);
    int64_t bestRounds = legacyRounds;
    int64_t bestGroups = 0;
    // This complete legacy stage retains its original candidate and tie order.
    for (const int64_t groups : GqaBandGroupRange(b, k, true)) {
        BuildLegacyGqaBandCandidate(cache, b, groups, plan);
        GqaBandAdmission admission;
        admission.queryLocality = true;
        admission.originalLocalityTie = true;
        TrySaveGqaBandCandidate(params, plan, legacyKvRuns, bestRounds, bestGroups, admission);
    }
    // Preserve every already admitted plan. If all minimum-round geometries
    // failed admission, try existing uniform geometries with the same guards.
    if (bestGroups == 0) {
        // Limit this second pass to forward general and paired-edge families.
        // Reversed and compound variants need a stronger locality admission model.
        constexpr GqaBandGeometryMode fallbackModes[] = {GqaBandGeometryMode::GENERAL,
                                                         GqaBandGeometryMode::PAIRED_EDGES};
        for (const int64_t groups : GqaBandGroupRange(b, k, true)) {
            for (const GqaBandGeometryMode geometry : fallbackModes) {
                const int64_t mode = GqaBandGeometryIndex(geometry);
                if (!cache.HasGeometry(mode)) {
                    continue;
                }
                GqaBandPlan fallback;
                cache.Build(groups, mode, mode, false, fallback);
                GqaBandAdmission admission;
                admission.queryLocality = true;
                TrySaveGqaBandCandidate(params, fallback, legacyKvRuns, bestRounds, bestGroups, admission);
            }
        }
    }
    // A whole diagonal needs no cross-column dQ reduction. Keep each KV column
    // on one core across G heads, and replace an existing plan only at fewer rounds.
    if (p == 1 && q == 1) {
        plan = GqaBandPlan{};
        const int64_t diagonalSize = std::min(m, n);
        AppendGqaBandPhase(plan, GqaBandPhaseKind::DIAGONAL_COLUMN_OWNER, diagonalSize, diagonalSize, 0, 0);
        SetGqaBandGroups(k, b, params.g, b, plan);
        TrySaveGqaBandCandidate(params, plan, legacyKvRuns, bestRounds, bestGroups);
    }
    // Mirrored column pairs balance band edges and retain each column on one owner.
    // One lane per group offers no inter-lane balancing; keep existing mappings.
    if (GqaBandColumnOwnerLanes(k, b, params.g) > 1 && CanUseGqaBandColumnOwner(k, b, params.g, m, n, p, q)) {
        plan = GqaBandPlan{};
        AppendGqaBandPhase(plan, GqaBandPhaseKind::BAND_COLUMN_OWNER, m, n, p, q);
        SetGqaBandGroups(k, b, params.g, b, plan);
        TrySaveGqaBandCandidate(params, plan, legacyKvRuns, bestRounds, bestGroups);
    }
    // Search complete chunks and their actual remainder with both column orders.
    // Preserve every existing tie; a new order/grouping must reduce total rounds.
    // A single lane may still remove masked work, so admission is based on cost.
    constexpr GqaBandPhaseKind ownerKinds[] = {GqaBandPhaseKind::BAND_COLUMN_OWNER,
                                               GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER};
    for (const int64_t groups : GqaBandGroupRange(b, k, false)) {
        for (const GqaBandPhaseKind kind : ownerKinds) {
            plan = GqaBandPlan{};
            AppendGqaBandPhase(plan, kind, m, n, p, q);
            SetGqaBandGroups(k, b, params.g, groups, plan);
            GqaBandAdmission admission;
            admission.completeChunks = true;
            TrySaveGqaBandCandidate(params, plan, legacyKvRuns, bestRounds, bestGroups, admission);
        }
    }
    SelectGqaBandFoldedOwner(params, k, m, n, p, q, b, 0, 0, legacyKvRuns, bestRounds, bestGroups);
    // Seed the new search with the complete previous selector, including its
    // even-height overlap candidate and original conservative cost admission.
    // Otherwise a new owner could block an older, faster overlap through its
    // tighter KV reference before that previous candidate was evaluated.
    if (m % NUM_TWO == 0) {
        SelectGqaBandOverlapSchedule(params, k, m, n, p, q, b, validColumns, legacyKvRuns, bestRounds, bestGroups);
    }
    // Refine only branches with a closed-form exact cost. Partial physical
    // tiles and unproved original mappings retain the conservative bound.
    const int64_t legacyExact = params.s1 % cubeBase == 0 && params.s2 % cubeBase == 0 ?
                                    GqaBandLegacyKvRunExact(k, m, n, p, q, b, params.g) :
                                    -1;
    const int64_t legacyKvLimit = legacyExact >= 0 ? legacyExact : legacyKvRuns;
    SelectGqaBandJointSchedule(params, k, m, n, q, b, validColumns, legacyKvLimit, cache, bestRounds, bestGroups);
    SelectGqaBandGenericFoldedOwner(params, k, m, n, p, q, b, validColumns, legacyKvLimit, bestRounds, bestGroups);
    SelectGqaBandOverlapSchedule(params, k, m, n, p, q, b, validColumns, legacyKvLimit, bestRounds, bestGroups);
    if (bestGroups != 0) {
        params.deterGqaBandScheduleMode = DeterGqaBandScheduleMode::PIECEWISE;
        params.deterGqaBandGroups = static_cast<uint32_t>(bestGroups);
        params.deterMaxRound = bestRounds;
    }
}
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_BAND_SELECTOR_H_
