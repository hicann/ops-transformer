/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GROUP_MEMCPY_TILING_POLICY_H
#define GROUP_MEMCPY_TILING_POLICY_H
#include <cstdint>
#include <limits>

namespace group_memcpy {
// Keep the original single-buffer, two-queue, 184 KiB UB policy.
constexpr uint64_t UB_BUDGET = 184 * 1024;
constexpr uint64_t ALIGNMENT = 64;
inline uint64_t TileBytes(uint64_t width, uint64_t ubBytes)
{
    if (width == 0 || width > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - ALIGNMENT) {
        return 0;
    }
    const uint64_t budget = ubBytes < UB_BUDGET ? ubBytes : UB_BUDGET;
    const uint64_t maxWidth = budget / 2 / ALIGNMENT * ALIGNMENT;
    if (maxWidth == 0) {
        return 0;
    }
    const uint64_t parts = width / maxWidth + (width % maxWidth != 0);
    const uint64_t rawWidth = width / parts;
    return (rawWidth + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}
} // namespace group_memcpy
#endif
