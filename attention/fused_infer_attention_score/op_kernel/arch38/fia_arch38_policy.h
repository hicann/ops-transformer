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
 * \file fia_arch38_policy.h
 * \brief arch38 platform policies shared by op_host and op_kernel.
 */

#ifndef FIA_ARCH38_POLICY_H
#define FIA_ARCH38_POLICY_H

#include <cstdint>

namespace FiaPolicy {
// dV 模板桶，与 kernel 侧 regbaseutil::DTemplateType 语义一致（host/kernel 共用）
constexpr uint32_t DV_TEMPLATE_64 = 64;
constexpr uint32_t DV_TEMPLATE_128 = 128;
constexpr uint32_t DV_TEMPLATE_192 = 192;
constexpr uint32_t DV_TEMPLATE_256 = 256;
constexpr uint32_t DV_TEMPLATE_512 = 512;

// Force-Bmm2-to-UB 模板触发条件：atten mask + dV 模板桶 256 + S1 基本块 64。
// dVBucket 传模板桶值（64/128/256/512），与 kernel 侧 (uint32_t)dVTemplateType 语义一致；
// host 侧须传桶值（dVBasicBlock），不可传原始 head size。
constexpr uint32_t FORCE_BMM2_UB_S1_SIZE = 64;

inline constexpr bool IsForceBmm2UB(bool hasAttenMask, uint32_t dVBucket, uint32_t s1BaseSize)
{
    return hasAttenMask && (dVBucket == DV_TEMPLATE_256) && (s1BaseSize == FORCE_BMM2_UB_S1_SIZE);
}
} // namespace FiaPolicy

#endif // FIA_ARCH38_POLICY_H
