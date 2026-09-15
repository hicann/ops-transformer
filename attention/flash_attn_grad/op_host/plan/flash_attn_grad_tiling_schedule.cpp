/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_schedule.h"

#include "flash_attn_grad_block_outer.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus BuildSchedulePlan(gert::TilingContext *context, const FagParsedInfo &info, const FagKernelPlan &kernel,
                                  FagSchedulePlan &schedule)
{
    const auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 启动的 cube 数 = dense 非 TND 下 metadata 的 blockOuter。tiling 读不到
    // AICPU 的张量，所以这里的公式必须与 DoFagBn2DenseSplit / DoFagDenseSplit
    // 保持一致（共用 flash_attn_grad_block_outer.h）。
    // sparse / TND / seqused 实际用的核可能比 dense 公式少，这些场景一律保持
    // 全片启动，确保永不少发。
    FagS1S2Outer(info.s1, info.s2, schedule.s1Outer, schedule.s2Outer);
    schedule.useActualCores = !info.isTnd && (info.maskMode == MASK_MODE_NO_MASK) && !info.hasSeqused;
    schedule.usedCube = info.aicNum;
    if (schedule.useActualCores) {
        schedule.usedCube =
            static_cast<uint32_t>(FagUsedCubeCores(kernel.tmpl == TMPL_BN2, info.b, info.n2, info.g, schedule.s1Outer,
                                                   schedule.s2Outer, static_cast<int64_t>(info.aicNum)));
    }

    schedule.blockDim =
        ascendcPlatform.CalcTschBlockDim(schedule.usedCube * FAG_AICV_RATIO_DEFAULT, info.aicNum, info.aivNum);
    OP_CHECK_IF(schedule.blockDim == 0,
                OP_LOGE(context->GetNodeName(), "CalcTschBlockDim returned 0, usedCube=%u, aicNum=%u, aivNum=%u.",
                        schedule.usedCube, info.aicNum, info.aivNum),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
