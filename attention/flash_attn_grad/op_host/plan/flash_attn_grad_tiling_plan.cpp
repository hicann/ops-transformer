/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_plan.h"

#include "flash_attn_grad_tiling_key.h"
#include "flash_attn_grad_tiling_route.h"
#include "flash_attn_grad_tiling_schedule.h"
#include "flash_attn_grad_tiling_swizzle.h"
#include "flash_attn_grad_tiling_workspace.h"
#include "log/log.h"

namespace optiling {
namespace {

constexpr int64_t FAG_PRESFMG_MIN_BLOCK_NUM = 1024;

// 判定是否把 sfmg 前移到 pre 阶段，判据对齐 fasg DoPreTiling 的 enablePreSfmg
// （arch35 flash_attention_score_grad_tiling_normal_regbase.cpp 的 fp16/bf16 分支）。
//
//   fasg enablePreSfmg = deterSupport && (
//       hifp8 || (fp16/bf16 && presfmgLimit && 64 < d <= 768 &&
//                 splitAxis in {BN2GS1S2, BN2S2} && noSink &&
//                 dropout % 8 == 0 && !sValueZeroUnderTND))
//
// FAG 的映射与差异（均为输入域差异，不是判据放宽）：
//   * 只有 fp16/bf16，没有 fp8 分支；也没有 dropout 输入，dropout 子句恒真。
//   * splitAxis：FAG 只有 BN2GS1S2 对应 fasg 的 BN2GS1S2；BN2/BN2MultiBlk 对应
//     fasg 的 BN2，不启用（FAG 无 BN2S2 路径）。
//   * d 用 Dv：sfmg 的归约宽度是 value dim；fasg 的判据用 d 是以 d==d1 为前提，
//     而 FAG 允许 D=192/Dv=128，故取实际 Dv。
//   * sValueZeroUnderTND：fasg 用于保护其按 (b,n,s) 逐批推进的索引；FAG 的 pre
//     行集合直接由 packed token 总数构造，零长序列不产生行，该限制不适用。
//   * deterSupport：FAG 的确定性路径即 fasg 的 new-deter（DENSE/CAUSAL/BAND），
//     恒为真。
//   * presfmgLimit = !(d <= 128 && s2 <= 256 && b*n1*s1Outer >= 1024)：小 D、
//     小 S2 且块数多时前移收益不抵开销，改回主循环内联重算。
bool DecidePreSfmg(const FagParsedInfo& info, const FagKernelPlan& kernel, const FagSchedulePlan& schedule)
{
    if (kernel.tmpl != TMPL_BN2GS1S2) {
        return false;
    }
    if (info.hasSinks) {
        return false;
    }
    if (info.dv <= 64 || info.dv > MAX_HEAD_DIM || kernel.dvAlign < 128) {
        return false;
    }
    const bool smallShapeNotWorth =
        info.dv <= 128 && info.s2 <= 256 && info.b * info.n1 * schedule.s1Outer >= FAG_PRESFMG_MIN_BLOCK_NUM;
    return !smallShapeNotWorth;
}

} // namespace

ge::graphStatus BuildTilingPlan(gert::TilingContext* context, const FagParsedInfo& info, FagTilingPlan& plan,
                                FagRouteTrace& trace)
{
    // 顺序不可调换：
    //   route    产出 tmpl，schedule 要用它区分 BN2 与 BN2GS1S2 的 fusedOuter 口径；
    //   schedule 产出 usedCube，swizzle 要用它判断是否满片启动。
    ge::graphStatus ret = SelectKernelPlan(context, info, plan.kernel, trace);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = BuildSchedulePlan(context, info, plan.kernel, plan.schedule);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DecideSwizzle(info, plan.kernel, plan.schedule, trace);

    // sfmg 前移门（对齐 fasg enablePreSfmg）：与 swizzle 无关，只依赖
    // 已完成的路由（template/isBn2MultiBlk）与 schedule（s1Outer）。
    plan.kernel.enablePreSfmg = DecidePreSfmg(info, plan.kernel, plan.schedule) ? 1 : 0;

    plan.workspace = BuildWorkspacePlan(info);
    plan.tilingKey = EncodeTilingKey(info, plan.kernel, plan.schedule);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
