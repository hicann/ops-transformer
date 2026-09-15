/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_key.h"

namespace optiling {
namespace {

constexpr uint64_t KEY_SHIFT_LAYOUT = 2;
constexpr uint64_t KEY_SHIFT_MASK_MODE = 3;
constexpr uint64_t KEY_SHIFT_SWIZZLE = 5;
constexpr uint64_t KEY_SHIFT_D_ALIGN = 6;
constexpr uint64_t KEY_SHIFT_DV_ALIGN = 8;
constexpr uint64_t KEY_SHIFT_BN2_MULTIBLK = 9;
constexpr uint64_t KEY_SHIFT_BN2_NEED_ZERO = 10;

uint64_t DAlignBits(int64_t dAlign)
{
    if (dAlign == 64) {
        return 0ULL;
    }
    return (dAlign == 128) ? 1ULL : 2ULL;
}

} // namespace

uint64_t EncodeTilingKey(const FagParsedInfo &info, const FagKernelPlan &kernel, const FagSchedulePlan &schedule)
{
    uint64_t maskModeBits = 0ULL;
    if (info.hasAttenMask && info.maskMode == MASK_MODE_CAUSAL) {
        maskModeBits = 1ULL;
    } else if (info.hasAttenMask && info.maskMode == MASK_MODE_WINDOW) {
        maskModeBits = 2ULL;
    }
    const uint64_t dvAlignBit = (kernel.dvAlign == 192) ? 1ULL : 0ULL;
    return static_cast<uint64_t>(kernel.tmpl) | (static_cast<uint64_t>(info.layout) << KEY_SHIFT_LAYOUT) |
           (maskModeBits << KEY_SHIFT_MASK_MODE) | (static_cast<uint64_t>(schedule.swizzle) << KEY_SHIFT_SWIZZLE) |
           (DAlignBits(kernel.dAlign) << KEY_SHIFT_D_ALIGN) | (dvAlignBit << KEY_SHIFT_DV_ALIGN) |
           (static_cast<uint64_t>(kernel.isBn2MultiBlk) << KEY_SHIFT_BN2_MULTIBLK) |
           (static_cast<uint64_t>(kernel.bn2NeedZero) << KEY_SHIFT_BN2_NEED_ZERO);
}

} // namespace optiling
