/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FAG_DETERMINISTIC_FIXED_BASELINE_H_
#define FAG_DETERMINISTIC_FIXED_BASELINE_H_

#include "../plan_types.h"
#include "candidate_types.h"
#include "fag_deter_utils.h"
#include <algorithm>

namespace optiling {
namespace fag_pto_fixed {

// These bounds belong to the source decoder, not to mask_mode. In particular,
// rectangular right-down causal inputs use the generic BAND dispatcher.
// Preserve physical round zero and every empty slot; no prefix trimming or
// additional pipeline drain rounds are part of this scheduling plan.
inline int64_t FixedMhaBandBaselineRounds(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                          int64_t sourceHostRounds)
{
    using fag_pto_arch35::CeilDivideBy;
    const int64_t tailRounds =
        p + q > m ? m * CeilDivideBy(n * b, std::min(k, b * m)) : CeilDivideBy(n * b, k) * (p + q - 1);
    return std::max(sourceHostRounds, tailRounds);
}

inline int64_t FixedGqaBandBaselineRounds(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g)
{
    using fag_pto_arch35::CeilDivideBy;
    // This is GenGQABandInfo's rm + rm2. Its local geometry adjustment must
    // not be committed to the plan: the kernel reconstructs the same segments.
    if (p + q > m) {
        const int64_t l1 = m - p;
        const int64_t l2 = p + q - m;
        const int64_t l3 = std::min(m - 1, n - q);
        n = l1 + l2 + l3;
    } else {
        const int64_t l1 = q - 1;
        const int64_t l2 = std::min(n - q + 1, m + 2 - p - q);
        const int64_t l3 = std::max<int64_t>(0, std::min(p + n - m - 1, p + q - 2));
        if (l3 == 0) {
            m = p + q + l2 - 2;
        }
        n = l1 + l2 + l3;
    }
    const int64_t tiles = m * n - (m - p) * (m - p + 1) / 2 - (n - q) * (n - q + 1) / 2;
    const int64_t fullRounds = (b / k) * tiles * g;
    const int64_t tailBatches = b % k;
    const auto denseRounds = [=](int64_t rows, int64_t columns) {
        return rows * std::max({CeilDivideBy(tailBatches * columns * g, k), CeilDivideBy(columns, rows), g});
    };
    int64_t tailRounds = 0;
    if (p + q > m) {
        tailRounds = denseRounds(m, p + q <= n ? p + q - 1 : n);
    } else if (n > m) {
        tailRounds = std::min(denseRounds(p + q - 1, n), denseRounds(m, p + q - 1));
    } else {
        tailRounds = denseRounds(p + q - 1, n);
    }
    return fullRounds + tailRounds;
}

inline void CompleteFixedBaselinePlan(const fag_pto_arch35::FixedCandidateState& candidate, FagDeterPlan& out)
{
    using namespace fag_pto_arch35;
    const int64_t b = candidate.b * candidate.n2;
    const int64_t k = candidate.aicNum;
    if (candidate.deterSparseType == static_cast<int64_t>(DeterSparseType::DETER_DENSE)) {
        out.family = DeterFamily::FIXED_MHA_DENSE_BASELINE;
        out.maxRound = CeilDivideBy(out.n * b, std::min(k, out.m * b)) * out.m;
    } else if (candidate.deterSparseType == static_cast<int64_t>(DeterSparseType::DETER_CAUSAL)) {
        out.family = DeterFamily::FIXED_MHA_CAUSAL_BASELINE;
        out.maxRound = candidate.deterMaxRound;
    } else if (candidate.g == 1) {
        out.family = DeterFamily::FIXED_MHA_BAND_BASELINE;
        out.maxRound = FixedMhaBandBaselineRounds(k, out.m, out.n, out.p, out.q, b, candidate.deterMaxRound);
    } else {
        out.family = DeterFamily::FIXED_GQA_BAND_BASELINE;
        // The source chooses k before translating the sparse geometry.
        out.band.k = std::min({k, b * candidate.g * candidate.s1Outer, b * candidate.s2Outer});
        out.maxRound = FixedGqaBandBaselineRounds(out.band.k, out.m, out.n, out.p, out.q, b, candidate.g);
    }
}

} // namespace fag_pto_fixed
} // namespace optiling
#endif
