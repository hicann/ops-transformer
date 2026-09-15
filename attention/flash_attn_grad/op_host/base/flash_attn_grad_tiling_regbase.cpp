/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_regbase.h"

#include <cmath>
#include <cstddef>

#include "../info/flash_attn_grad_tiling_info_parser.h" // IsEmptyShape：只用于日志
#include "../plan/flash_attn_grad_tiling_plan.h"
#include "op_host/tiling_util.h"

namespace optiling {
namespace {

// TilingData 有两份互不相干的声明，谁都不引用谁：
//   1. op_kernel/flash_attn_grad.py 的 FlashAttnGradTilingData dataclass 是权威，
//      codegen 按它的声明顺序生成 FlashAttnGradTilingData_tiling.h（本文件强制包含），
//      也就是下面真正写入的那份内存布局；
//   2. op_host/flash_attn_grad_tiling.h（入口旁）的 FlashAttnGradTilingDataTmp 是手写的，
//      只为 REGISTER_TILING_DATA_CLASS 让框架知道该分配多大。
// 两份漂移不会有任何编译或运行报错，只会让 kernel 读到整体错位的字段。下面把
// 布局钉死：改了 py 侧字段就会在这里断编译，逼着同步改 Tmp 和这些数字。
// 注意 scaleValue 是唯一的 4 字节字段，它后面有 4 字节填充 —— 框架的
// CheckAlignAndGenPlaceHolder 也会插同样的填充，两边才对得上。
static_assert(sizeof(FlashAttnGradTilingData) == 192,
              "TilingData layout changed; update FlashAttnGradTilingDataTmp in flash_attn_grad_tiling.h too.");
static_assert(offsetof(FlashAttnGradTilingData, scaleValue) == 56, "scaleValue moved; re-check the 4-byte padding.");
static_assert(offsetof(FlashAttnGradTilingData, gSize) == 64, "gSize moved; re-check the padding after scaleValue.");
static_assert(offsetof(FlashAttnGradTilingData, coefN2) == 128,
              "layout-view tail moved; a field was added or removed.");
static_assert(offsetof(FlashAttnGradTilingData, totalPerBatchNum) == 184,
              "trailing mask field moved; a field was added or removed.");

void LogPlan(gert::TilingContext *context, const FagParsedInfo &info, const FagTilingPlan &plan,
             const FagRouteTrace &trace)
{
    OP_LOGI(context->GetNodeName(),
            "FlashAttnGrad tiling: usedCube=%u, usedBlockDim=%u, useActualCores=%d, s1Outer=%ld, s2Outer=%ld, "
            "layoutStr=%s, layout=%ld, maskMode=%ld, win=(%ld,%ld), sparseType=%ld, totalPerBatch=%ld, "
            "hasAttenMask=%d, hasSeqused=%d, hasSinks=%d, isEmptyShape=%d, "
            "template=%ld, isBn2MultiBlk=%ld, bn2NeedZero=%ld, swizzle=%ld, dAlign=%ld, dvAlign=%ld, "
            "B=%ld, S1=%ld, S2=%ld, N1=%ld, N2=%ld, G=%ld, D=%ld, Dv=%ld, "
            "viewQ=[%ld,S,%ld,D] b0=%ld, viewKV=[%ld,S,%ld,D] b0=%ld, coefN=(%ld,%ld), "
            "tilingKey=%llu, workspaceSize=%zu.",
            plan.schedule.usedCube, plan.schedule.blockDim, static_cast<int>(plan.schedule.useActualCores),
            plan.schedule.s1Outer, plan.schedule.s2Outer, info.layoutQ.c_str(), info.layout, info.maskMode,
            info.winLeft, info.winRight, info.sparseType, info.totalPerBatchNum, static_cast<int>(info.hasAttenMask),
            static_cast<int>(info.hasSeqused), static_cast<int>(info.hasSinks), static_cast<int>(IsEmptyShape(info)),
            plan.kernel.tmpl, plan.kernel.isBn2MultiBlk, plan.kernel.bn2NeedZero, plan.schedule.swizzle,
            plan.kernel.dAlign, plan.kernel.dvAlign, info.b, info.s1, info.s2, info.n1, info.n2, info.g, info.d,
            info.dv, info.viewD0Q, info.viewD2Q, info.coefB0Q, info.viewD0KV, info.viewD2KV, info.coefB0KV, info.coefN0,
            info.coefN2, plan.tilingKey, plan.workspace.totalSize);

    // 决策追溯：上板发现别的模板更快时，先看这行就知道今天为什么选了它。
    // l2CacheSize 一起打出来：swizzle 的阈值以它为分母，各 ascend950 变体
    // 16MB~128MB 不等，判据对不上时第一件事就是核对这个数。
    OP_LOGI(context->GetNodeName(), "FlashAttnGrad route: chosen=%s, swizzle=%s, aicNum=%u, l2CacheSize=%lu.",
            trace.chosen, trace.swizzleReject, info.aicNum, info.l2CacheSize);
    for (size_t i = 0; i < trace.rejectedNum; ++i) {
        OP_LOGI(context->GetNodeName(), "FlashAttnGrad route: rejected %s by %s.", trace.rejectedName[i],
                trace.rejectedReason[i]);
    }
}

} // namespace

void FlashAttnGradTilingRegbase::InitTilingInfo(const FagParsedInfo &info)
{
    info_ = &info;
}

bool FlashAttnGradTilingRegbase::IsCapable()
{
    // 唯一的能力边界。返回 false 让 DoTiling 交出 GRAPH_PARAM_INVALID，语义是
    // "本类不接，去试下一个模板类"；今天没有下一个，框架会据此报不支持。
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        OP_LOGE(context_->GetNodeName(), "SOC Version is not support.");
        return false;
    }
    return true;
}

ge::graphStatus FlashAttnGradTilingRegbase::DoOpTiling()
{
    const FagParsedInfo &info = *info_;

    FagTilingPlan plan;
    FagRouteTrace trace;
    ge::graphStatus ret = BuildTilingPlan(context_, info, plan, trace);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = SetNumBlocks(plan.schedule.blockDim);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // BN2GS1S2 的 pre/post 用 sync_all；BN2 仅 bn2NeedZero 那份二进制有
    // sync_all。batch mode 在未满核启动时必须开，否则可能卡死。
    if (plan.kernel.tmpl == TMPL_BN2GS1S2 || plan.kernel.bn2NeedZero == 1) {
        ret = SetScheduleMode(FagScheduleMode::BATCH_MODE);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    ret = WriteTilingData(info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = SetTilingKey(plan.tilingKey);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = SetWorkspaceSize(plan.workspace.totalSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    LogPlan(context_, info, plan, trace);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnGradTilingRegbase::WriteTilingData(const FagParsedInfo &info)
{
    FlashAttnGradTilingData *tilingData = context_->GetTilingData<FlashAttnGradTilingData>();
    // GetTilingData 在 capacity < sizeof(T) 时返回 nullptr，而 capacity 来自上面
    // 说的 Tmp。所以这里判空等价于「两份声明已经漂移」，不判就是段错误。
    OP_CHECK_IF(tilingData == nullptr,
                OP_LOGE(context_->GetNodeName(),
                        "tiling data buffer is smaller than %zu bytes; FlashAttnGradTilingDataTmp in "
                        "flash_attn_grad_tiling.h is out of sync with the pypto kernel's tiling data.",
                        sizeof(FlashAttnGradTilingData)),
                return ge::GRAPH_FAILED);

    tilingData->b = info.b;
    tilingData->s1 = info.s1;
    tilingData->s2 = info.s2;
    tilingData->n1 = info.n1;
    tilingData->n2 = info.n2;
    tilingData->d = info.d;
    tilingData->dv = info.dv;
    tilingData->scaleValue = info.scaleValue == 0.0f ? 1.0f / sqrt(static_cast<float>(info.d)) : info.scaleValue;
    tilingData->gSize = info.g;
    tilingData->viewD0Q = info.viewD0Q;
    tilingData->viewD2Q = info.viewD2Q;
    tilingData->coefB0Q = info.coefB0Q;
    tilingData->viewD0KV = info.viewD0KV;
    tilingData->viewD2KV = info.viewD2KV;
    tilingData->coefB0KV = info.coefB0KV;
    tilingData->coefN0 = info.coefN0;
    tilingData->coefN2 = info.coefN2;
    tilingData->maskMode = info.maskMode;
    tilingData->winLeft = info.winLeft;
    tilingData->winRight = info.winRight;
    tilingData->sparseType = info.sparseType;
    tilingData->s1Token = info.s1Token;
    tilingData->s2Token = info.s2Token;
    tilingData->totalPerBatchNum = info.totalPerBatchNum;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
