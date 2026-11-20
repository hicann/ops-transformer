/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYOUT_ARCH92_H
#define MEGA_MOE_LAYOUT_ARCH92_H

#if __has_include("../../arch35_92/common/mega_moe_gm_layout.h")
#include "../../arch35_92/common/mega_moe_gm_layout.h"
#else
#include "mega_moe_gm_layout.h"
#endif

#ifndef MEGA_MOE_ENABLE_ARCH92_ROW_COMBINE
#define MEGA_MOE_ENABLE_ARCH92_ROW_COMBINE 1
#endif
#ifndef MEGA_MOE_ENABLE_ARCH92_4K_PACKET
#define MEGA_MOE_ENABLE_ARCH92_4K_PACKET 1
#endif

#if (MEGA_MOE_ENABLE_ARCH92_ROW_COMBINE != 0 && MEGA_MOE_ENABLE_ARCH92_ROW_COMBINE != 1) || \
    (MEGA_MOE_ENABLE_ARCH92_4K_PACKET != 0 && MEGA_MOE_ENABLE_ARCH92_4K_PACKET != 1)
#error "Arch92 Combine switches must be 0 or 1"
#endif

namespace MegaMoeImpl {

namespace CommPacketArch92 {
constexpr uint32_t ROW_MODE = MEGA_MOE_ENABLE_ARCH92_ROW_COMBINE ? (MEGA_MOE_ENABLE_ARCH92_4K_PACKET ? 2U : 1U) : 0U;
constexpr uint32_t COMBINE_ROW_ALIGNMENT_BYTES = 4096U;

constexpr uint32_t GetCombineRowAlignmentBytes(bool isMte, bool quantized)
{
    return ROW_MODE != 0U && isMte && !quantized ? COMBINE_ROW_ALIGNMENT_BYTES : 0U;
}
} // namespace CommPacketArch92

namespace LayoutArch92 {

HOST_DEVICE constexpr bool NeedsExpertGmm2Ready(const CclComputeLayoutParams& params, bool alignCombineRows)
{
    return NeedsMteExpertGmm2Ready(params) || alignCombineRows;
}

HOST_DEVICE constexpr void SetTensorStorage(CclComputeLayoutParams& params, uint8_t moeGmmMode, uint8_t sharedGmmMode)
{
    params.dispatchFp4 = IsA4W4GmmMode(moeGmmMode);
    params.sharedInputElementsPerByte = IsA4W4GmmMode(sharedGmmMode) ? 2U : 1U;
    params.dispatchInputStorage = {params.dispatchFp4 ? MemoryLayoutDetail::FP4_BITS : MemoryLayoutDetail::FP8_BITS};
    params.sharedInputStorage = {IsA4W4GmmMode(sharedGmmMode) ? MemoryLayoutDetail::FP4_BITS :
                                                                MemoryLayoutDetail::FP8_BITS};
    // A4W4 still supplies FP8 activations to the native FP8 x FP4 GMM2.
    params.activationOutputStorage = {MemoryLayoutDetail::FP8_BITS};
    params.sharedActivationStorage = {MemoryLayoutDetail::FP8_BITS};
}

HOST_DEVICE constexpr WorkspaceLayout MakeWorkspaceLayout(CclComputeLayoutParams params, uint8_t moeGmmMode,
                                                          uint8_t sharedGmmMode, bool alignCombineRows = false)
{
    params.ring = {};
    params.reserveGmm1Output =
        params.topoType == TOPO_TYPE_MTE || moeGmmMode == GMM_MODE_A8W4_NZ || params.topkWeightsPrefetch;
    params.reserveCumsum = false;
    SetTensorStorage(params, moeGmmMode, sharedGmmMode);
    return WorkspaceLayout(params, {NeedsExpertGmm2Ready(params, alignCombineRows), /*useLayeredUrmaLayout=*/false});
}

} // namespace LayoutArch92
} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYOUT_ARCH92_H
