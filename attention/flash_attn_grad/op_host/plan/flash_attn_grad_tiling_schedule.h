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
 * \file flash_attn_grad_tiling_schedule.h
 * \brief L4 策略层：决定启动多少 cube 核（blockDim）。
 *
 * 分核区间本身不在 host —— 它由 AICPU metadata 产出，kernel 运行期读取。
 * host 只需要保证启动核数与 metadata 的 blockOuter 对得上，且遵守
 * "允许多发、禁止少发" 的单向安全（多发的核 start==end 自然 no-op）。
 */

#ifndef FLASH_ATTN_GRAD_TILING_SCHEDULE_H_
#define FLASH_ATTN_GRAD_TILING_SCHEDULE_H_

#include "exe_graph/runtime/tiling_context.h"
#include "../info/flash_attn_grad_tiling_info.h"

namespace optiling {

// 前置条件：kernel（模板选择）必须已经定下来，因为 BN2 与 BN2GS1S2 的
// fusedOuter 口径不同。
ge::graphStatus BuildSchedulePlan(gert::TilingContext *context, const FagParsedInfo &info, const FagKernelPlan &kernel,
                                  FagSchedulePlan &schedule);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_SCHEDULE_H_
