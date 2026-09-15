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
 * \file flash_attn_grad_tiling_swizzle.h
 * \brief L4 策略层：分核方式（线性均分 vs 按 s2 列跨核发放）。
 *
 * 这是一个纯偏好判据 —— 两种分核都算得对，只是快慢不同。判据必须保持
 * 纯整数、无浮点、无平台调用。kernel 只消费 tilingkey 里的 swizzle 位，
 * 不再维护 Python 镜像。
 */

#ifndef FLASH_ATTN_GRAD_TILING_SWIZZLE_H_
#define FLASH_ATTN_GRAD_TILING_SWIZZLE_H_

#include "../info/flash_attn_grad_tiling_info.h"

namespace optiling {

// 前置条件：kernel.tmpl 与 schedule.usedCube 必须已经定下来 —— swizzle 只在
// BN2GS1S2 且满片启动时才可能开。调用顺序不可与 BuildSchedulePlan 对调。
void DecideSwizzle(const FagParsedInfo &info, const FagKernelPlan &kernel, FagSchedulePlan &schedule,
                   FagRouteTrace &trace);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_SWIZZLE_H_
