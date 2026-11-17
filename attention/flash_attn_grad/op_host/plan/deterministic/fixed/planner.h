/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FAG_DETERMINISTIC_FIXED_PLANNER_H_
#define FAG_DETERMINISTIC_FIXED_PLANNER_H_
// PTO-owned pure planner shared by Host and the TND AICPU normalizer.
#include "../../../info/flash_attn_grad_tiling_info.h"
#include "candidate_types.h"
#include "fag_deter_utils.h"
#include "dense.h"
#include "causal.h"
#include "band/selector.h"
#include "band/grouped.h"
#include "baseline.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <type_traits>
namespace optiling {
namespace fag_pto_arch35 {

inline BlockScheduleResult SelectBlockSchedule(const FixedCandidateState& params, bool enableGrouped = true)
{
    BlockScheduleResult result;
    const bool canSwizzle = params.enableSwizzle && params.layoutType != INPUT_FORMAT_TND &&
                            params.splitAxis == SplitAxisEnum::BN2GS1S2 &&
                            params.s1Inner * params.s1CvRatio == params.s2Inner * params.s2CvRatio &&
                            params.sparseType != static_cast<uint8_t>(SparseType::UNSUPPORTED);
    result.isSwizzle = canSwizzle;
    if (!params.isDeterministic) {
        return result;
    }

    if (params.g > 1 && (params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) ||
                         params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL))) {
        const int64_t cubeBase = static_cast<int64_t>(ConstAxisTemplateNum::NUM128);
        // The common swizzle gate controls every GQA BAND/CAUSAL candidate. Candidate
        // admission remains a later decision and can retain the original mapping.
        result.isSwizzle = canSwizzle && params.aicNum > 0 && params.aicNum <= MAX_CORE_NUM &&
                           params.coreNum == params.aicNum * NUM_TWO && params.b > params.tailZeroCount &&
                           params.n2 > 0 && params.s1Outer > 0 && params.s2Outer > 0 &&
                           params.s1Inner * params.s1CvRatio == cubeBase &&
                           params.s2Inner * params.s2CvRatio == cubeBase;
        return result;
    }

    const bool causalCond = params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL);
    const bool rightDownBandCond = params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) &&
                                   params.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
    result.isSwizzle = canSwizzle && (((params.b * params.n2) & 1) == 0) && params.g == 1 &&
                       params.s1 >= params.aicNum * static_cast<uint32_t>(ConstAxisTemplateNum::NUM128) &&
                       (params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE) || causalCond ||
                        rightDownBandCond);

    // Preserve the legacy RIGHT_DOWN_CAUSAL swizzle whenever its original entry conditions are met.
    // With schedule mode DISABLED, the kernel keeps using CalCausalSwizzleIndex and its matching
    // max-round formula. Re-selecting this path as a hybrid DENSE schedule can increase the rounds.
    if (rightDownBandCond && result.isSwizzle) {
        return result;
    }

    const int64_t actualBatch = (params.b - params.tailZeroCount) * params.n2;
    // DETER_BAND is the scheduling contract here. It covers BAND, compatible NO_MASK, and
    // unequal RIGHT_DOWN_CAUSAL after their tokens have been normalized to the same band geometry.
    const bool hybridBandCond = canSwizzle &&
                                params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) &&
                                params.g == 1 && params.coreNum == params.aicNum * NUM_TWO && actualBatch > 0;
    if (!hybridBandCond) {
        return result;
    }

    const int64_t cubeBase = params.s1Inner * params.s1CvRatio;
    const int64_t serializedS1Token = std::min<int64_t>(params.s1Token, INT32_MAX);
    const int64_t serializedS2Token = std::min<int64_t>(params.s2Token, INT32_MAX);
    const int64_t p = CeilDivideBy(serializedS1Token, cubeBase) + 1;
    const int64_t q = CeilDivideBy(serializedS2Token, cubeBase) + 1;
    const DeterBandScheduleParams scheduleParams =
        NormalizeDeterBandScheduleParams(params.s1Outer, params.s2Outer, p, q);
    result.deterBandSchedule =
        SelectDeterBandSchedule(static_cast<int64_t>(params.aicNum), scheduleParams.m, scheduleParams.n,
                                scheduleParams.p, scheduleParams.q, actualBatch, params.deterMaxRound);
    if (enableGrouped && result.deterBandSchedule.mode == DeterBandScheduleMode::DISABLED &&
        params.sparseMode == static_cast<uint32_t>(SparseMode::BAND) && params.hasAttenMask &&
        cubeBase == static_cast<int64_t>(ConstAxisTemplateNum::NUM128)) {
        // PTO currently accepts FP16/BF16, without dropout, PSE or RoPE. Keep the
        // upstream physical-window and complete-tail cost checks in the pure planner.
        const BandPhysicalWindow window{params.s1,         params.s2,      serializedS1Token,
                                        serializedS2Token, q < 0 ? -q : 0, p < 0 ? -p : 0};
        std::vector<int64_t> firstRows;
        const auto cache = CalcBandCacheFootprint(params.aicNum, params.s1, params.s2, params.d, params.dv,
                                                  params.inputBytes, 2, false, 2048 * 2048, params.l2CacheSize);
        const auto heights =
            CalcBandPhysicalColumnHeights(params.aicNum, scheduleParams.m, scheduleParams.n, scheduleParams.p,
                                          scheduleParams.q, actualBatch, window, &firstRows);
        if (!heights.empty()) {
            result.deterBandSchedule = SelectGroupedDeterBandSchedule(params.aicNum, scheduleParams.m, scheduleParams.n,
                                                                      scheduleParams.p, scheduleParams.q, actualBatch,
                                                                      params.deterMaxRound, heights, firstRows, cache);
        }
    }
    if (result.deterBandSchedule.mode != DeterBandScheduleMode::DISABLED) {
        result.isSwizzle = true;
    }
    return result;
}

} // namespace fag_pto_arch35

namespace fag_pto_fixed {
using namespace fag_pto_arch35;
constexpr int64_t kTile = 128;
inline int64_t FixedDenseBlockOuter(int64_t total, int64_t cores)
{
    if (total <= 0 || cores <= 0) {
        return 0;
    }
    const int64_t factor = (total + cores - 1) / cores;
    return (total + factor - 1) / factor;
}

// Compute the exact union of row/column tile payloads visited by the baseline
// CheckExceedL2Cache. Its set grows throughout the traversal, so only the union
// matters. Interval accounting avoids a host loop over every tile and core.
struct FixedFootprint {
    int64_t validTiles = 0;
    int64_t queryRows = 0;
    int64_t kvColumns = 0;
};
inline FixedFootprint CountFixedFootprint(const FagParsedInfo& info, int64_t m, int64_t n)
{
    if (info.maskMode == MASK_MODE_NO_MASK) {
        return {m * n, m, n};
    }
    FixedFootprint result;
    // The split counts exact token intersections; the L2 walk uses its own
    // 64-aligned validity predicate. They differ on partial window edges.
    for (int64_t col = 0; col < n; ++col) {
        const int64_t beginToken = std::max<int64_t>(0, col * kTile - info.s2Token);
        const int64_t endToken =
            std::min(info.s1, std::max<int64_t>(0, std::min(info.s2, (col + 1) * kTile) + info.s1Token));
        const int64_t begin = beginToken > info.s1 ? m : beginToken / std::min(kTile, info.s1);
        const int64_t end = std::min(m, (endToken + kTile - 1) / kTile);
        result.validTiles += std::max<int64_t>(0, end - begin);
    }
    int64_t unionEnd = 0;
    for (int64_t row = 0; row < m; ++row) {
        int64_t left = 0;
        int64_t right = 0;
        if (info.maskMode == MASK_MODE_CAUSAL) {
            right = std::max<int64_t>(0, info.s2 - info.s1 + (row + 1) * kTile);
        } else {
            left = std::max<int64_t>(0, row * kTile - info.s1Token);
            left = (left + 63) / 64 * 64;
            right = (std::min((row + 1) * kTile, info.s1) + info.s2Token + 63) / 64 * 64;
        }
        right = std::min(right, info.s2);
        if (left >= info.s2) {
            continue;
        }
        const int64_t begin = std::max<int64_t>(0, std::min(n, left / kTile));
        const int64_t end = std::max<int64_t>(0, std::min(n, (right + kTile - 1) / kTile));
        if (end > begin) {
            ++result.queryRows;
            result.kvColumns += std::max<int64_t>(0, end - std::max(begin, unionEnd));
            unionEnd = std::max(unionEnd, end);
        }
    }
    return result;
}
inline bool EnableFixedSwizzle(const FagParsedInfo& info, const FixedFootprint& footprint)
{
    const __int128 total = static_cast<__int128>(footprint.validTiles) * info.b * info.n2 * info.g;
    if (total <= 0 || total > std::numeric_limits<int64_t>::max() || info.aicNum == 0) {
        return false;
    }
    if (FixedDenseBlockOuter(static_cast<int64_t>(total), info.aicNum) != info.aicNum) {
        return false;
    }
    // Current PTO input domain is FP16/BF16. Match AscendC's 128-row tile
    // payload accounting, including padded tails and its compressed mask.
    const __int128 q = static_cast<__int128>(footprint.queryRows) * info.b * info.n2 * info.g * kTile * info.d * 10;
    const __int128 kv = static_cast<__int128>(footprint.kvColumns) * info.b * info.n2 * kTile * info.d * 12;
    const int64_t maskBytes = info.maskMode == MASK_MODE_NO_MASK ? 0 : 2048 * 2048;
    return q + kv + maskBytes > info.l2CacheSize;
}

inline FixedCandidateState MakeFixedCandidate(const FagParsedInfo& info, const FagSchedulePlan& schedule)
{
    FixedCandidateState p;
    p.aicNum = info.aicNum;
    p.d = info.d;
    p.dv = info.dv;
    p.l2CacheSize = info.l2CacheSize;
    p.hasAttenMask = info.hasAttenMask;
    p.coreNum = info.aivNum;
    p.b = info.b;
    p.n2 = info.n2;
    p.g = info.g;
    p.s1 = info.s1;
    p.s2 = info.s2;
    p.s1Outer = schedule.s1Outer;
    p.s2Outer = schedule.s2Outer;
    p.s1Inner = 64;
    p.s1CvRatio = 2;
    p.s2Inner = 128;
    p.s2CvRatio = 1;
    p.s1Token = info.s1Token;
    p.s2Token = info.s2Token;
    p.sparseMode = info.maskMode;
    p.isDeterministic = true;
    // AscendC SetSparseParams treats a BAND whose normalized window covers
    // both sequences as dense. RIGHT_DOWN_CAUSAL remains sparse even when the
    // physical grid is rectangular or contains only one tile.
    const bool fullBand = info.maskMode == MASK_MODE_WINDOW && info.s1Token >= info.s1 && info.s2Token >= info.s2;
    p.deterSparseType = info.maskMode == MASK_MODE_NO_MASK || fullBand ?
                            2 :
                            (info.maskMode == MASK_MODE_CAUSAL && info.s1 == info.s2 ? 3 : 4);
    p.enableSwizzle = EnableFixedSwizzle(info, CountFixedFootprint(info, p.s1Outer, p.s2Outer));
    return p;
}

inline void SetBandGeometry(const FixedCandidateState& p, FagDeterPlan& out)
{
    out.m = p.s1Outer;
    out.n = p.s2Outer;
    out.p = std::min(out.m, CeilDivideBy(std::min<int64_t>(p.s1Token, INT32_MAX), kTile) + 1);
    out.q = std::min(out.n, CeilDivideBy(std::min<int64_t>(p.s2Token, INT32_MAX), kTile) + 1);
    if (out.p < 0) {
        out.colOffset = -out.p;
        out.n += out.p;
        out.q += out.p;
        out.p = 1;
    } else if (out.q < 0) {
        out.rowOffset = -out.q;
        out.m += out.q;
        out.p += out.q;
        out.q = 1;
    }
}
static_assert(std::extent<decltype(FagDeterBandPlan::phases)>::value == GQA_BAND_MAX_PHASES &&
                  std::extent<decltype(FagDeterBandPlan::tailPhases)>::value == GQA_BAND_MAX_PHASES,
              "Fixed result and BAND schedules must have matching phase capacities");

inline void CopyBandSchedule(const GqaBandSchedule& in, FagDeterBandPlan& out)
{
    out.phaseCount = in.phaseCount;
    out.tailPhaseCount = in.tailPhaseCount;
    out.k = in.k;
    out.b = in.b;
    out.g = in.g;
    out.groupsPerChunk = in.groupsPerChunk;
    out.fullRounds = in.fullRounds;
    out.tailRounds = in.tailRounds;
    out.totalRounds = in.totalRounds;
    for (int64_t i = 0; i < GQA_BAND_MAX_PHASES; ++i) {
        const auto copy = [](const GqaBandPhase& a, FagDeterPhase& b) {
            b = {static_cast<int64_t>(a.kind), a.m, a.n, a.rowOffset, a.colOffset, a.fullRounds, a.tailRounds};
        };
        copy(in.phases[i], out.phases[i]);
        copy(in.tailPhases[i], out.tailPhases[i]);
    }
}
inline int64_t BandFamily(const GqaBandSchedule& plan)
{
    if (plan.phaseCount != 1 || plan.tailPhaseCount != 0) {
        return DeterFamily::FIXED_GQA_BAND_PIECEWISE;
    }
    const auto& phase = plan.phases[0];
    switch (phase.kind) {
        case GqaBandPhaseKind::BAND_COLUMN_OWNER:
            return DeterFamily::FIXED_GQA_BAND_COLUMN;
        case GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER:
            return DeterFamily::FIXED_GQA_BAND_BALANCED;
        case GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER:
            return phase.m == phase.n ? DeterFamily::FIXED_GQA_BAND_TRANSLATED : DeterFamily::FIXED_GQA_BAND_PIECEWISE;
        case GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER:
        case GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER:
        case GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER:
            return DeterFamily::FIXED_GQA_BAND_FOLDED;
        case GqaBandPhaseKind::OVERLAP_LOWER:
        case GqaBandPhaseKind::OVERLAP_UPPER:
            return DeterFamily::FIXED_GQA_BAND_OVERLAP;
        default:
            return DeterFamily::FIXED_GQA_BAND_PIECEWISE;
    }
}

inline FagDeterPlan BuildFixedCandidate(const FagParsedInfo& info, const FagSchedulePlan& schedule, int64_t& swizzle,
                                        bool enableGrouped = true)
{
    FixedCandidateState p = MakeFixedCandidate(info, schedule);
    FagDeterPlan out;
    out.sparseKind = p.deterSparseType;
    out.m = p.s1Outer;
    out.n = p.s2Outer;
    const int64_t b = info.b * info.n2;
    const int64_t k = info.aicNum;
    if (p.deterSparseType == 2) {
        if (info.g > 1) {
            const auto dense = SelectGQADenseSchedule(k, out.m, out.n, b, info.g);
            out.family = DeterFamily::FIXED_GQA_DENSE;
            out.denseRound = dense.selectedRound;
            out.maxRound = dense.selectedRound * out.m;
        } else {
            p.deterMaxRound = PositiveCeilDiv(out.n * b, std::min(k, out.m * b)) * out.m;
            const auto selected = SelectBlockSchedule(p, enableGrouped);
            swizzle = selected.isSwizzle;
            if (swizzle != 0) {
                out.family = DeterFamily::FIXED_MHA_DENSE;
                out.maxRound = p.deterMaxRound;
            }
        }
    } else if (p.deterSparseType == 3) {
        CalcleCausalDeterParam(p, false);
        if (info.g > 1) {
            if (enableGrouped) {
                const bool selected = CalcleCausalDeterParam(p, b > 1 && SelectBlockSchedule(p).isSwizzle);
                swizzle = selected ? 1 : 0;
                out.family = selected ? DeterFamily::FIXED_GQA_CAUSAL_GROUPED : DeterFamily::FIXED_GQA_CAUSAL;
            } else {
                // TND metadata keeps its existing normalization policy and wire format.
                CalcleNormalizedCausalDeterParam(p, true);
                swizzle = 0;
                out.family = DeterFamily::FIXED_GQA_CAUSAL;
            }
            out.causalRoundsPerGroup = p.deterGqaCausalRoundsPerGroup;
            out.causalGroups = p.deterGqaCausalGroups;
            out.maxRound = p.deterMaxRound;
        } else {
            swizzle = SelectBlockSchedule(p, enableGrouped).isSwizzle;
            if (swizzle != 0) {
                out.family = DeterFamily::FIXED_MHA_CAUSAL;
                out.maxRound = std::max(out.m * PositiveCeilDiv((out.n + 1) * (b / 2), k), out.n + 1);
            }
        }
    } else {
        CalcleBandDeterParam(p);
        SetBandGeometry(p, out);
        const auto selected = SelectBlockSchedule(p, enableGrouped);
        swizzle = selected.isSwizzle;
        p.isSwizzle = swizzle;
        if (info.g > 1 && swizzle != 0) {
            SelectGqaBandSchedule(p, true);
            if (p.deterGqaBandScheduleMode == DeterGqaBandScheduleMode::PIECEWISE) {
                out.family = BandFamily(p.deterGqaBandSchedule);
                out.maxRound = p.deterMaxRound;
                CopyBandSchedule(p.deterGqaBandSchedule, out.band);
            } else {
                // Generic GQA BAND has its own small compiled family. It keeps
                // the source dispatcher even when no optimized phase is admitted.
                swizzle = 0;
            }
        } else if (info.g == 1 && swizzle != 0) {
            out.family = selected.deterBandSchedule.mode == DeterBandScheduleMode::GROUPED ?
                             DeterFamily::FIXED_MHA_BAND_GROUPED :
                             DeterFamily::FIXED_MHA_BAND;
            if (selected.deterBandSchedule.mode == DeterBandScheduleMode::GROUPED) {
                CopyBandSchedule(selected.deterBandSchedule.groupedSchedule, out.band);
            }
            out.bandMode = static_cast<int64_t>(selected.deterBandSchedule.mode);
            out.maxRound = selected.deterBandSchedule.maxRound;
            if (selected.deterBandSchedule.mode == DeterBandScheduleMode::DISABLED) {
                // Legacy right-down swizzle is selected before hybrid candidates.
                if (p.s1Outer < p.s2Outer) {
                    out.m = p.s1Outer;
                    out.n = p.s2Outer;
                    out.rowOffset = 0;
                    out.colOffset = 0;
                    const int64_t virtualN = 2 * out.n - out.m + 3;
                    out.maxRound = std::max(out.m * PositiveCeilDiv(virtualN * (b / 2), k), virtualN);
                } else {
                    out.rowOffset = p.s1Outer - p.s2Outer - 1;
                    out.colOffset = 0;
                    out.m = p.s2Outer + 1;
                    out.n = out.m;
                    out.maxRound = std::max(out.m * PositiveCeilDiv((out.n + 1) * (b / 2), k), out.n + 1);
                }
            }
        }
    }
    if (out.family == DeterFamily::LEGACY) {
        CompleteFixedBaselinePlan(p, out);
        swizzle = 0;
    }
    return out;
}
} // namespace fag_pto_fixed
} // namespace optiling
#endif
