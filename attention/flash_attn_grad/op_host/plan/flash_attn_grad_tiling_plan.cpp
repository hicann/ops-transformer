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

ge::graphStatus BuildTilingPlan(gert::TilingContext *context, const FagParsedInfo &info, FagTilingPlan &plan,
                                FagRouteTrace &trace)
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

    plan.workspace = BuildWorkspacePlan(info);
    plan.tilingKey = EncodeTilingKey(info, plan.kernel, plan.schedule);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
