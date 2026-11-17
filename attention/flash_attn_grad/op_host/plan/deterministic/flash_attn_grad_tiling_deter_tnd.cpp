/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_deter_tnd.h"
#include "family.h"

namespace optiling {
void DecideTndDeterministicSchedule(const FagParsedInfo& info, const FagKernelPlan& kernel, FagSchedulePlan& schedule,
                                    FagRouteTrace& trace)
{
    if (!info.isDeterministic || !info.isTnd || kernel.tmpl != TMPL_BN2GS1S2 || info.aicNum == 0 || info.aicNum > 36 ||
        info.aivNum != 2 * info.aicNum || info.b <= 0 || info.n2 <= 0 || info.g <= 0) {
        trace.swizzleReject = "tnd-deterministic-fallback-capability";
        return;
    }
    // With no effective-length override, equal physical totals prove Sq == Sk
    // whenever AICPU confirms every batch has the same respective lengths.
    // This is a compile-time capability proof, not a sequence-value readback.
    const bool normalizedSquare = !info.hasSeqused && info.t1 == info.t2;
    int64_t family = DeterFamily::LEGACY;
    if (info.maskMode == MASK_MODE_NO_MASK) {
        family = info.g == 1 ? DeterFamily::TND_MHA_DENSE : DeterFamily::TND_GQA_DENSE;
    } else if (info.maskMode == MASK_MODE_WINDOW) {
        // Actual lengths may be shorter than public maxima. This family must
        // include the reference's actual-length Dense normalization.
        family = info.g == 1 ? DeterFamily::TND_MHA_SPARSE : DeterFamily::TND_GQA_DENSE;
    } else if (info.g == 1 && info.maskMode == MASK_MODE_CAUSAL) {
        family = DeterFamily::TND_MHA_SPARSE;
    } else if (info.maskMode == MASK_MODE_CAUSAL && info.g > 1) {
        if (info.g % 4 == 0 && info.n2 == 1) {
            family = normalizedSquare ? DeterFamily::TND_GQA_MIRROR : DeterFamily::TND_GQA_MIRROR_OR_BAND;
        } else if (info.g % 4 == 0 && info.n2 % 2 == 0 && info.isBf16 && info.d == info.dv &&
                   (info.d == 128 || info.d == 256)) {
            family = normalizedSquare ? DeterFamily::TND_GQA_COLUMN_C4 : DeterFamily::TND_GQA_COLUMN_C4_OR_BAND;
        } else {
            family = normalizedSquare ? DeterFamily::TND_GQA_CAUSAL_COMPAT : DeterFamily::TND_GQA_CAUSAL_OR_BAND_COMPAT;
        }
    }
    if (family == DeterFamily::LEGACY) {
        trace.swizzleReject = "tnd-deterministic-static-fallback";
        return;
    }
    // No sequence values are copied to Host. A candidate key is not evidence
    // that its dynamic admission passed: protocolStatus identifies that result.
    schedule.deter.family = family;
    schedule.swizzle = family == DeterFamily::TND_MHA_DENSE ? 1 : 0;
    trace.swizzleReject = "tnd-deterministic-metadata-candidate";
}
} // namespace optiling
