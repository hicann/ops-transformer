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
 * \file flash_attn_grad_tiling_route.h
 * \brief L4 策略层：选 kernel 模板（tilingkey 的 template / is_bn2_multiblk 两个字段）。
 *
 * 这是随上板实测变动最频繁的一段：
 *   - 候选按性能优先级排成一张表，逐行试；
 *   - 每个候选的**能力**（能不能跑对）与**偏好**（哪个更快）是两个独立谓词，
 *     改性能标定只动 prefer，永不碰 feasible；
 *   - 每条淘汰都写进 trace，让"今天为什么选了它"可以从日志直接读出来。
 *
 * 函数签名刻意用 const 入参 + 返回值：路由不得修改解析结果。
 */

#ifndef FLASH_ATTN_GRAD_TILING_ROUTE_H_
#define FLASH_ATTN_GRAD_TILING_ROUTE_H_

#include "exe_graph/runtime/tiling_context.h"
#include "../info/flash_attn_grad_tiling_info.h"

namespace optiling {

// 选模板 + 定 D/Dv 分档。失败仅有一种情况：D 超出已实现范围。
ge::graphStatus SelectKernelPlan(gert::TilingContext *context, const FagParsedInfo &info, FagKernelPlan &plan,
                                 FagRouteTrace &trace);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_ROUTE_H_
