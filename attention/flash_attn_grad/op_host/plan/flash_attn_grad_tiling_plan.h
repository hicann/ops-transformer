/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_grad_tiling_plan.h
 * \brief L5 编排层：按固定顺序调用 L4 各策略，产出完整的 FagTilingPlan。
 *
 * 策略之间存在顺序依赖，且是隐式的（swizzle 要读 usedCube、schedule 要读
 * tmpl），所以编排集中在一处，不散落到入口函数里。
 */

#ifndef FLASH_ATTN_GRAD_TILING_PLAN_H_
#define FLASH_ATTN_GRAD_TILING_PLAN_H_

#include "exe_graph/runtime/tiling_context.h"
#include "../info/flash_attn_grad_tiling_info.h"

namespace optiling {

// 当前 TND 与非 TND 共用这一条编排：两者的差异（实际序列长度）全部由 AICPU
// metadata 承担，host 侧只有 tilingkey 的 layout 位不同。
//
// 预留扩展点：待 TND 专用流水（varlen 模板）落地后，在 BuildTilingPlan 内按
// info.isTnd 分派到独立的 BuildVarlenTilingPlan，届时本函数保持非 TND 语义。
// 现在不提前拆，是因为拆了会立刻产生一条永远走不到的死分支。
ge::graphStatus BuildTilingPlan(gert::TilingContext *context, const FagParsedInfo &info, FagTilingPlan &plan,
                                FagRouteTrace &trace);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_PLAN_H_
