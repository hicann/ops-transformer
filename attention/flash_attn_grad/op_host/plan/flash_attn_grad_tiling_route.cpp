/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_route.h"

#include "flash_attn_grad_sparse.h"
#include "log/log.h"

namespace optiling {
namespace {

// ---------------------------------------------------------------------------
// 能力边界常量：这些数字来自 kernel 实现边界，不是性能标定，改动必须伴随 kernel 改动
// ---------------------------------------------------------------------------
constexpr int64_t BN2_MAX_S = 128;        // 单块 BN2 的 S 上限
constexpr int64_t BN2_MAX_D = 128;        // BN2 一系列模板的 D 上限
constexpr int64_t BN2_MULTIBLK_SEQ = 640; // MultiBlk 的 S 上限

// ---------------------------------------------------------------------------
// 偏好门限常量：全部来自上板实测，改这里不影响功能正确性，只影响快慢
// ---------------------------------------------------------------------------
constexpr int64_t BN2_MULTIBLK_BN_128 = 128;
constexpr int64_t BN2_MULTIBLK_BN_256 = 256;
constexpr int64_t BN2_ALIGN128 = 128;

// ===== 能力谓词 =============================================================
// 共同前提：BN2 系列模板本轮都不接 TND、不接 GQA、不接 D>128。
bool Bn2FamilyFeasible(const FagParsedInfo &p)
{
    return !p.isTnd && p.n1 == p.n2 && p.d <= BN2_MAX_D;
}

bool Feasible_Bn2Single(const FagParsedInfo &p)
{
    return Bn2FamilyFeasible(p) && p.s1 <= BN2_MAX_S && p.s2 <= BN2_MAX_S;
}

bool Feasible_Bn2MultiBlk(const FagParsedInfo &p)
{
    // S 落在 (128, 640] 且 D==Dv：这些都是 MultiBlk kernel 的实现前提。
    // mask 3/4 也可以走 MultiBlk；无效行/列由 bn2NeedZero 另开一份二进制。
    return Bn2FamilyFeasible(p) && (p.s1 > BN2_MAX_S || p.s2 > BN2_MAX_S) && p.s1 <= BN2_MULTIBLK_SEQ &&
           p.s2 <= BN2_MULTIBLK_SEQ && p.d == p.dv;
}

// D5：BN2S2 是保留位，本轮不实现不编译，host 永不产生 template=3。
bool Feasible_Never(const FagParsedInfo &)
{
    return false;
}

bool Feasible_Always(const FagParsedInfo &)
{
    return true;
}

// ===== 偏好谓词 =============================================================
bool Prefer_Always(const FagParsedInfo &)
{
    return true;
}

// BN 够大时 MultiBlk 才划算：BN>=256，或 BN>=128 且 S1/S2 都按 128 对齐。
bool Prefer_Bn2BnGate(const FagParsedInfo &p)
{
    const int64_t bn = p.b * p.n1;
    return (bn >= BN2_MULTIBLK_BN_256) ||
           (bn >= BN2_MULTIBLK_BN_128 && (p.s1 % BN2_ALIGN128 == 0) && (p.s2 % BN2_ALIGN128 == 0));
}

// ===== 候选表 ===============================================================
// 顺序即性能优先级，越靠前越希望命中；末行必须恒可行，保证一定选得出来。
// 新增 kernel 模板 = 加一行；实测结论变化 = 调顺序或改 prefer。
struct FagRoute {
    const char *name;
    int64_t tmpl;
    int64_t isBn2MultiBlk;
    bool (*feasible)(const FagParsedInfo &);
    bool (*prefer)(const FagParsedInfo &);
    const char *calibration; // 该行偏好门限的依据出处
};

constexpr FagRoute kRoutes[] = {
    {"BN2_single", TMPL_BN2, 0, Feasible_Bn2Single, Prefer_Always,
     "S<=128 时不存在多核累加，省掉 pre 清零与 post cast"},
    {"BN2_multiblk", TMPL_BN2, 1, Feasible_Bn2MultiBlk, Prefer_Bn2BnGate, "BN>=256，或 BN>=128 且 S1/S2 对齐 128"},
    {"BN2S2_reserved", TMPL_RESERVED_BN2S2, 0, Feasible_Never, Prefer_Always,
     "D5 保留位：本轮不实现不编译，勿复用该 template 取值"},
    {"GS1S2", TMPL_BN2GS1S2, 0, Feasible_Always, Prefer_Always, "兜底，恒可行"},
};

constexpr size_t kRouteNum = sizeof(kRoutes) / sizeof(kRoutes[0]);
static_assert(kRouteNum <= FAG_TRACE_SLOT_NUM, "trace slots must cover all routes");

// ===== D / Dv 分档 ==========================================================
// D 决定每块片上 tile 的宽度，而 tile shape 在 pypto 是 trace 期常量，所以必须
// 做成编译期分档。一律按最大 192 分配不只是
// 慢：L0C 会要 288KB，超出 256KB 预算，物理上放不下。
// Dv 只决定已分配 tile 里填多少列，是纯运行期数值，不占 tilingkey；但 y/dy/prod/tmp
// 四块 UB 的宽度由 Dv 派生，故仍需一个分配档次。
ge::graphStatus SelectDBucket(gert::TilingContext *context, const FagParsedInfo &p, FagKernelPlan &plan)
{
    OP_CHECK_IF(p.d > MAX_HEAD_DIM,
                OP_LOGE(context->GetNodeName(), "head dim of q (%ld) is not supported; only D <= %ld is implemented.",
                        p.d, MAX_HEAD_DIM),
                return ge::GRAPH_PARAM_INVALID);

    plan.dAlign = (p.d <= 64) ? 64 : ((p.d <= 128) ? 128 : 192);
    plan.dvAlign = (p.dv <= 128) ? 128 : 192;
    // dAlign=64 仍是独立的 tilingkey 档，但 kernel 的 L1/L0 物理上保持 128 宽、
    // 靠 compact + set_validshape 裁剪；dvAlign 没有 64 档。
    if (plan.dAlign == 64) {
        plan.dvAlign = 128;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus SelectKernelPlan(gert::TilingContext *context, const FagParsedInfo &info, FagKernelPlan &plan,
                                 FagRouteTrace &trace)
{
    bool matched = false;
    for (size_t i = 0; i < kRouteNum; ++i) {
        const FagRoute &route = kRoutes[i];
        if (!route.feasible(info)) {
            trace.Reject(route.name, "capability");
            continue;
        }
        if (!route.prefer(info)) {
            trace.Reject(route.name, "preference");
            continue;
        }
        plan.tmpl = route.tmpl;
        plan.isBn2MultiBlk = route.isBn2MultiBlk;
        trace.chosen = route.name;
        matched = true;
        break;
    }

    // 末行恒可行，正常不会走到这里；留断言防止有人删掉兜底行。
    OP_CHECK_IF(!matched, OP_LOGE(context->GetNodeName(), "no feasible route found, the fallback route is missing."),
                return ge::GRAPH_FAILED);
    // D5 兜底：保留取值绝不能流到 tilingkey。
    OP_CHECK_IF(plan.tmpl != TMPL_BN2GS1S2 && plan.tmpl != TMPL_BN2,
                OP_LOGE(context->GetNodeName(), "route produced reserved template %ld, only %ld/%ld are implemented.",
                        plan.tmpl, TMPL_BN2GS1S2, TMPL_BN2),
                return ge::GRAPH_FAILED);

    if (plan.isBn2MultiBlk == 1 && (info.maskMode == MASK_MODE_CAUSAL || info.maskMode == MASK_MODE_WINDOW) &&
        FagBn2HasInvalidOuter(info.s1, info.s2, info.s1Token, info.s2Token)) {
        plan.bn2NeedZero = 1;
    }

    return SelectDBucket(context, info, plan);
}

} // namespace optiling
