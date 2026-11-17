/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_deter.h"
#include "fixed/planner.h"
namespace optiling {
using namespace fag_pto_fixed;
bool IsFixedDeterministicFamily(int64_t family)
{
    return (family >= DeterFamily::FIXED_MHA_DENSE && family <= DeterFamily::FIXED_GQA_BAND_OVERLAP) ||
           (family >= DeterFamily::FIXED_MHA_DENSE_BASELINE && family <= DeterFamily::FIXED_MHA_BAND_GROUPED);
}

void DecideFixedDeterministicSchedule(const FagParsedInfo& info, const FagKernelPlan& kernel, FagSchedulePlan& schedule,
                                      FagRouteTrace& trace)
{
    if (!info.isDeterministic || info.isTnd || kernel.tmpl != TMPL_BN2GS1S2 || info.hasSeqused || info.aicNum == 0 ||
        info.aicNum > 36 || info.aivNum != 2 * info.aicNum || info.b <= 0 || info.n2 <= 0 || info.g <= 0 ||
        schedule.s1Outer <= 0 || schedule.s2Outer <= 0) {
        trace.swizzleReject = "deterministic-fallback-capability";
        return;
    }
    // Keep products and the baseline decoder's R+1 drain representable.
    __int128 product = static_cast<__int128>(info.b) * info.n2 * info.g * schedule.s1Outer * schedule.s2Outer;
    if (product > std::numeric_limits<int64_t>::max() / 64) {
        trace.swizzleReject = "deterministic-round-capacity";
        return;
    }
    if (CountFixedFootprint(info, schedule.s1Outer, schedule.s2Outer).validTiles == 0) {
        trace.swizzleReject = "deterministic-empty-window";
        return;
    }
    FagSchedulePlan candidate = schedule;
    candidate.swizzle = 0;
    candidate.deter = BuildFixedCandidate(info, schedule, candidate.swizzle);
    if (candidate.deter.maxRound >= std::numeric_limits<uint32_t>::max()) {
        trace.swizzleReject = "deterministic-round-capacity";
        return;
    }
    candidate.deterMaxRound = candidate.deter.maxRound;
    schedule = candidate;
    trace.swizzleReject = schedule.swizzle != 0 ? "accepted" : "deterministic-nonswizzle";
}
} // namespace optiling
