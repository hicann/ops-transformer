/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_CAUSAL_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_CAUSAL_H_

// Fixed causal round bounds and grouped candidate selection.
#include <algorithm>
#include <limits>
#include "candidate_types.h"
#include "fag_deter_utils.h"
#include "causal_grouped.h"

namespace optiling {
namespace fag_pto_arch35 {
inline int64_t CalcCausalSingleBatchRound(int64_t k, int64_t causalSize)
{
    if (k <= 0 || causalSize <= 0) {
        return 0;
    }
    // Keep this bound identical to CalCausalSingleBatchDeterIndex. Every complete 2*k-column
    // group consumes a shrinking rectangle; the remaining at most 2*k columns form one tail.
    const int64_t groupCount = causalSize / (NUM_TWO * k);
    const int64_t groupRound = (NUM_TWO * causalSize + 1) * groupCount - NUM_TWO * k * groupCount * groupCount;
    const int64_t remain = causalSize - NUM_TWO * k * groupCount;
    if (remain <= k) {
        return groupRound + remain;
    }
    return groupRound + std::max(remain, NUM_TWO * remain - NUM_TWO * k + 1);
}

inline int64_t CalcCausalDeterRound(int64_t k, int64_t m, int64_t n, int64_t b, int64_t g)
{
    if (k <= 0 || m <= 0 || n <= 0 || m < n || b <= 0 || g <= 0) {
        return 0;
    }

    int64_t bTail = b % k;
    int64_t rUpper = b / k * (n * m - n * (n - 1) / MULT_BASE);
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t t1 = n / (MULT_BASE * k);
    int64_t n1 = t * k;

    if (g != 1) {
        rUpper += (MULT_BASE * m - n1 + 1) * t * (bTail / MULT_BASE);
    } else {
        rUpper += bTail * (n1 * m - n1 * (n1 - 1) / MULT_BASE) / k;
    }
    if (bTail % MULT_BASE == 1) {
        if ((t % MULT_BASE) == 1) {
            int64_t m1 = m - t1 * MULT_BASE * k;
            if (ell == 0) {
                int64_t rm3 = (g != 1) ? (m + m1 + 1) * t1 : 0;
                rUpper += m1 + rm3;
            } else {
                int64_t rm3 = (g != 1) ? (m + m1 + 1) * t1 : 0;
                rUpper += std::max(m1, MULT_BASE * m1 - MULT_BASE * k + 1) + rm3;
            }
            bTail = bTail - 1;
        } else {
            rUpper += (NUM_TWO * m - n1 + 1) * t / MULT_BASE;
        }
    }

    int64_t ell1, L;
    if (ell % MULT_BASE == 0) {
        ell1 = ell / MULT_BASE;
        L = MULT_BASE * (m - n) + ell + 1;
    } else {
        ell1 = ell / MULT_BASE + 1;
        L = MULT_BASE * (m - n) + ell;
    }
    rUpper += CeilDivideBy(ell1 * bTail, k) * L;
    return rUpper * g;
}

inline bool NormalizeCausalTileGeometry(FixedCandidateState& fBaseParams, int64_t& m, int64_t& n)
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) && m > n) {
        int64_t skipM = (fBaseParams.s1 - fBaseParams.s2) / (fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        m -= skipM;
    } else if ((fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) &&
               n > m) {
        n = m;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) && m < n) {
        fBaseParams.deterSparseType = static_cast<uint32_t>(DeterSparseType::DETER_BAND);
        return false;
    }
    return true;
}

inline bool CalcleCausalDeterParam(FixedCandidateState& fBaseParams, bool enableGrouped)
{
    fBaseParams.deterGqaCausalRoundsPerGroup = 0;
    fBaseParams.deterGqaCausalGroups = 0;
    int64_t m{fBaseParams.s1Outer}, n{fBaseParams.s2Outer}, k{static_cast<int64_t>(fBaseParams.aicNum)},
        b{fBaseParams.b * fBaseParams.n2};
    if (!NormalizeCausalTileGeometry(fBaseParams, m, n)) {
        return false;
    }

    const int64_t legacyRound = CalcCausalDeterRound(k, m, n, b, fBaseParams.g);
    fBaseParams.deterMaxRound = legacyRound;
    // The pipeline requests the next round while draining, so UINT32_MAX is not a valid bound.
    if (!enableGrouped || legacyRound <= 0 || legacyRound >= std::numeric_limits<uint32_t>::max()) {
        return fBaseParams.deterGqaCausalGroups != 0;
    }
    const int64_t originalGroups = std::min(b, k);
    const CausalWork originalFull = CalcCausalDeterWork(k, m, n, originalGroups, fBaseParams.g);
    const CausalWork originalTail = CalcCausalDeterWork(k, m, n, b % originalGroups, fBaseParams.g);
    if (!HasCompleteCausalWork(originalFull, m, n, originalGroups, fBaseParams.g) ||
        !HasCompleteCausalWork(originalTail, m, n, b % originalGroups, fBaseParams.g)) {
        // Some existing single-BN2 rectangular mappings do not cover the full
        // region. Retain the pre-existing H1/original selection for those inputs;
        // do not extend their use or treat omitted tasks as a cost improvement.
        const int64_t roundsPerBatch = CalcCausalDeterRound(k, m, n, 1, fBaseParams.g);
        if (roundsPerBatch > 0 && roundsPerBatch <= legacyRound / b) {
            fBaseParams.deterGqaCausalGroups = 1;
            fBaseParams.deterGqaCausalRoundsPerGroup = roundsPerBatch;
            fBaseParams.deterMaxRound = b * roundsPerBatch;
        }
        return fBaseParams.deterGqaCausalGroups != 0;
    }
    // The legacy floating inverse can omit the first rounds of complete 2K-column
    // parts in an odd BN2 tail. Use the same grouping with exact boundaries before
    // comparing costs, including when cache capacity is unknown. Complete K-sized
    // groups retain their original coordinates; the tail regains all valid tasks.
    const bool needsExactReference = (b % k) % NUM_TWO != 0 && n / k >= NUM_TWO;
    if (needsExactReference) {
        fBaseParams.deterGqaCausalGroups = originalGroups;
        fBaseParams.deterGqaCausalRoundsPerGroup = CalcCausalDeterRound(k, m, n, originalGroups, fBaseParams.g);
    }
    if (fBaseParams.l2CacheSize == 0) {
        return fBaseParams.deterGqaCausalGroups != 0;
    }
    const uint64_t bn2Bytes = EstimateCausalBn2Bytes(fBaseParams);
    if (bn2Bytes == std::numeric_limits<uint64_t>::max()) {
        return fBaseParams.deterGqaCausalGroups != 0;
    }
    // The original rm1 phase already advances in K-sized BN2 groups. Model that
    // existing locality instead of treating every BN2 in a large batch as live.
    const int64_t maxGroups = std::min<int64_t>(std::min(b, k), std::numeric_limits<uint8_t>::max());
    std::vector<CausalWork> groupWork;
    groupWork.reserve(maxGroups + 1);
    for (int64_t groups = 0; groups <= maxGroups; ++groups) {
        groupWork.push_back(CalcCausalDeterWork(k, m, n, groups, fBaseParams.g));
    }
    const CausalGroupCost original = EstimateCausalGroupCost(b, originalGroups, legacyRound, bn2Bytes,
                                                             fBaseParams.l2CacheSize, originalFull, originalTail);
    CausalGroupCost best = original;
    for (int64_t groups = 1; groups <= maxGroups; ++groups) {
        const int64_t fullGroupCount = b / groups;
        const int64_t tailGroups = b % groups;
        const int64_t fullRound = CalcCausalDeterRound(k, m, n, groups, fBaseParams.g);
        const int64_t tailRound = tailGroups == 0 ? 0 : CalcCausalDeterRound(k, m, n, tailGroups, fBaseParams.g);
        // Check before multiplying, and reject every candidate that increases rounds.
        if (fullRound <= 0 || tailRound < 0 || tailRound > legacyRound ||
            fullRound > (legacyRound - tailRound) / fullGroupCount) {
            continue;
        }
        if (!HasCompleteCausalWork(groupWork[groups], m, n, groups, fBaseParams.g) ||
            !HasCompleteCausalWork(groupWork[tailGroups], m, n, tailGroups, fBaseParams.g)) {
            continue;
        }
        const int64_t candidateRound = fullGroupCount * fullRound + tailRound;
        const auto candidate = EstimateCausalGroupCost(b, groups, candidateRound, bn2Bytes, fBaseParams.l2CacheSize,
                                                       groupWork[groups], groupWork[tailGroups]);
        // Do not turn inexpensive all-core empty rounds into additional useful
        // execution rounds, or concentrate more work on the busiest core. Compare
        // with the fixed original schedule, so eligibility is search-order independent.
        if (candidate.activeRounds > original.activeRounds || candidate.maxCoreTasks > original.maxCoreTasks ||
            !IsBetterCausalGroupCost(candidate, best)) {
            continue;
        }
        best = candidate;
        fBaseParams.deterGqaCausalGroups = groups;
        fBaseParams.deterGqaCausalRoundsPerGroup = fullRound;
        fBaseParams.deterMaxRound = candidateRound;
    }
    return fBaseParams.deterGqaCausalGroups != 0;
}
// The TND normalizer retains its established single-BN2 ordering. This compatibility
// selector is deliberately separate from the fixed-layout grouped optimization.
inline void CalcleNormalizedCausalDeterParam(FixedCandidateState& fBaseParams, bool enableBatchMajor)
{
    fBaseParams.deterGqaCausalGroups = 0;
    fBaseParams.deterGqaCausalRoundsPerGroup = 0;
    int64_t m{fBaseParams.s1Outer}, n{fBaseParams.s2Outer}, k{static_cast<int64_t>(fBaseParams.aicNum)},
        b{fBaseParams.b * fBaseParams.n2};
    if (!NormalizeCausalTileGeometry(fBaseParams, m, n)) {
        return;
    }

    const int64_t legacyRound = CalcCausalDeterRound(k, m, n, b, fBaseParams.g);
    fBaseParams.deterMaxRound = legacyRound;
    const int64_t cubeBase = static_cast<int64_t>(ConstAxisTemplateNum::NUM128);
    const bool supported = enableBatchMajor && fBaseParams.isDeterministic &&
                           fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL) &&
                           fBaseParams.layoutType != INPUT_FORMAT_TND &&
                           fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 && fBaseParams.g > 1 && b > 1 &&
                           fBaseParams.coreNum == k * NUM_TWO &&
                           fBaseParams.s1Inner * fBaseParams.s1CvRatio == cubeBase &&
                           fBaseParams.s2Inner * fBaseParams.s2CvRatio == cubeBase;
    // Process_NEW_DETER requests roundId + 1 while draining its last pipeline round.
    if (!supported || legacyRound <= 0 || legacyRound >= std::numeric_limits<uint32_t>::max()) {
        return;
    }

    const int64_t roundsPerBatch = CalcCausalDeterRound(k, m, n, 1, fBaseParams.g);
    const int64_t roundLimit = std::min<int64_t>(std::numeric_limits<uint32_t>::max() - 1, legacyRound);
    // Compare before multiplying so a rejected batch-major candidate cannot overflow.
    if (roundsPerBatch <= 0 || roundsPerBatch > roundLimit / b) {
        return;
    }
    fBaseParams.deterGqaCausalGroups = 1;
    fBaseParams.deterGqaCausalRoundsPerGroup = roundsPerBatch;
    fBaseParams.deterMaxRound = b * roundsPerBatch;
}
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_CAUSAL_H_
