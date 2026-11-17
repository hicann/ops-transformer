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
 * \file fa_arch38_policy.h
 * \brief arch38 platform policies shared by op_host and op_kernel.
 */

#ifndef FA_ARCH38_POLICY_H
#define FA_ARCH38_POLICY_H

#include <cstdint>

namespace ArchInfo {
// Force-BMM2-to-UB template condition: atten mask + dV template bucket 256 + S1 base block 64.
// dVBucket uses the template bucket value (64/128/256/512), matching the kernel-side
// (uint32_t)dVTemplateType semantics; the host side must pass the same bucket value (dVBasicBlock),
// not the raw head size.
constexpr uint32_t FORCE_BMM2_UB_DV_TEMPLATE = 256;
constexpr uint32_t FORCE_BMM2_UB_S1_SIZE = 64;

inline constexpr bool IsForceBmm2UB(bool hasAttenMask, uint32_t dVBucket, uint32_t s1BaseSize)
{
    return hasAttenMask && (dVBucket == FORCE_BMM2_UB_DV_TEMPLATE) && (s1BaseSize == FORCE_BMM2_UB_S1_SIZE);
}
} // namespace ArchInfo

#endif // FA_ARCH38_POLICY_H
