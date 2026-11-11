/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_nzout.h"

namespace optiling {
namespace {
constexpr int64_t NZ_OUT_MIN_D = 64;
constexpr int64_t NZ_OUT_MAX_D = 128;
constexpr int64_t NZ_OUT_D_ALIGNMENT = 16;
constexpr int64_t NZ_OUT_MIN_S = 2048;
} // namespace

bool NzOutFeasible(const FagParsedInfo& info, const FagKernelPlan& kernel)
{
    // The operator definition already restricts inputs to FP16/BF16.
    return kernel.tmpl == TMPL_BN2GS1S2 && kernel.isBn2MultiBlk == 0 && kernel.bn2NeedZero == 0 &&
           kernel.dAlign == NZ_OUT_MAX_D && kernel.dvAlign == NZ_OUT_MAX_D && info.d == info.dv &&
           info.d > NZ_OUT_MIN_D && info.d < NZ_OUT_MAX_D && info.d % NZ_OUT_D_ALIGNMENT != 0;
}

bool PreferNzOut(const FagParsedInfo& info)
{
    // Initial long-sequence preference; kept separate from functional support
    // so board A/B can calibrate it without changing the workspace contract.
    return info.s1 >= NZ_OUT_MIN_S && info.s2 >= NZ_OUT_MIN_S;
}

void DecideNzOut(const FagParsedInfo& info, FagKernelPlan& kernel, FagRouteTrace& trace)
{
    kernel.nzOut = 0;
    if (!NzOutFeasible(info, kernel)) {
        trace.nzOutReject = "capability";
        return;
    }
    if (!PreferNzOut(info)) {
        trace.nzOutReject = "short-sequence";
        return;
    }
    kernel.nzOut = 1;
    trace.nzOutReject = "accepted";
}

} // namespace optiling
