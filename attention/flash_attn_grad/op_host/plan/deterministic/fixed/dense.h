/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_DENSE_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_DENSE_H_

// Fixed GQA Dense candidate and its original locality admission.
#include <algorithm>
#include "candidate_types.h"
#include "fag_deter_utils.h"

namespace optiling {
namespace fag_pto_arch35 {
constexpr int64_t GQA_DENSE_PERCENT_BASE = 100, GQA_DENSE_MAX_ROUND_GROWTH_PERCENT = 3,
                  GQA_DENSE_MAX_INVALID_PERCENT = 4;

inline GQADenseScheduleResult SelectGQADenseSchedule(int64_t k, int64_t m, int64_t n, int64_t b, int64_t g)
{
    GQADenseScheduleResult result;
    if (k <= 0 || m <= 0 || n <= 0 || b <= 0 || g <= 1) {
        return result;
    }

    k = std::min({k, b * g * m, b * n});
    result.baseRound = std::max({PositiveCeilDiv(b * n * g, k), PositiveCeilDiv(n, m), g});
    result.selectedRound = result.baseRound;
    const int64_t batchHeadNum = b * g;
    result.basePeriod = batchHeadNum / PositiveGcd(batchHeadNum, result.baseRound);
    result.selectedPeriod = result.basePeriod;
    if (result.baseRound % g == 0) {
        return result;
    }

    const int64_t candidateRound = PositiveCeilDiv(result.baseRound, g) * g;
    const int64_t candidatePeriod = batchHeadNum / PositiveGcd(batchHeadNum, candidateRound);
    const int64_t totalId = b * n * g;
    const int64_t candidateInvalid = k * candidateRound - totalId;
    // Aligning R to g can shorten the batch/head traversal period. Only accept it when the extra
    // rounds and invalid IDs are bounded and m has enough row offsets for the new period.
    const bool roundCostOk = (candidateRound - result.baseRound) * GQA_DENSE_PERCENT_BASE <=
                             result.baseRound * GQA_DENSE_MAX_ROUND_GROWTH_PERCENT;
    const bool invalidCostOk =
        candidateInvalid * GQA_DENSE_PERCENT_BASE <= k * candidateRound * GQA_DENSE_MAX_INVALID_PERCENT;
    const bool localityBetter = candidatePeriod < result.basePeriod && candidatePeriod < k;
    const bool rowOffsetEnough = PositiveCeilDiv(k, candidatePeriod) <= m;
    if (roundCostOk && invalidCostOk && localityBetter && rowOffsetEnough) {
        result.selectedRound = candidateRound;
        result.selectedPeriod = candidatePeriod;
    }
    return result;
}
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_DENSE_H_
