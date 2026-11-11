/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_TILING_NZOUT_H_
#define FLASH_ATTN_GRAD_TILING_NZOUT_H_

#include "../info/flash_attn_grad_tiling_info.h"

namespace optiling {

// Selection happens after the existing route. NZ output never reroutes BN2.
// It is independent of swizzle and preserves the deterministic schedule.
bool NzOutFeasible(const FagParsedInfo& info, const FagKernelPlan& kernel);
bool PreferNzOut(const FagParsedInfo& info);
void DecideNzOut(const FagParsedInfo& info, FagKernelPlan& kernel, FagRouteTrace& trace);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_NZOUT_H_
