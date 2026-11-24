/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_QUANT_LAYOUT_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_QUANT_LAYOUT_H

#include <cstdint>

#if defined(__NPU_ARCH__)
#define MEGA_MHC_QUANT_LAYOUT_INLINE __aicore__ inline
#else
#define MEGA_MHC_QUANT_LAYOUT_INLINE inline
#endif

namespace MegaMhc::QuantOutput {
// arch35's quantizer reads 256 B16 values and 128 uint16 scale entries per physical step.
constexpr uint32_t kVectorBytes = 256U;
constexpr uint32_t kScaleVectorElements = kVectorBytes / sizeof(uint16_t);
// DINTLV loads two B16 registers per conversion step.
constexpr uint32_t kInputRegistersPerStep = 2U;
constexpr uint32_t kTileElements = kInputRegistersPerStep * kScaleVectorElements;
// OCP shares one E8M0 scale per group; the public scale tensor packs scales in pairs.
constexpr uint32_t kGroupElements = 32U;
constexpr uint32_t kScalesPerPair = 2U;
constexpr uint32_t kPairElements = kGroupElements * kScalesPerPair;
// Each output credit owns a scale slot and, when needed, a padded tail tile.
constexpr uint32_t kOutputSlots = 2U;
constexpr uint32_t kOutputSlotMask = kOutputSlots - 1U;
static_assert((kOutputSlots & kOutputSlotMask) == 0U, "Output slots must be a power of two");

struct ScratchLayout {
    uint32_t batchElements = 0U;
    uint32_t slotBytes = 0U;
    uint32_t inverseOffset = 0U;
    uint32_t tailOffset = 0U;
    uint32_t bytes = 0U;
};

/** Shared Host/Kernel allocation contract. Only a short final physical tile needs padded B16 storage. */
MEGA_MHC_QUANT_LAYOUT_INLINE ScratchLayout MakeScratchLayout(uint32_t maxElements, uint32_t lastElements)
{
    ScratchLayout layout;
    if (maxElements == 0U) {
        return layout;
    }
    layout.batchElements = maxElements / kTileElements * kTileElements;
    const uint32_t scales = layout.batchElements / kGroupElements;
    const uint32_t scaleVectors = (scales + kScaleVectorElements - 1U) / kScaleVectorElements;
    const bool needsTail = maxElements % kTileElements != 0U || lastElements % kTileElements != 0U;
    // Tail scales must not overwrite full-block scales still being read by MTE3.
    const uint32_t slotVectors = scaleVectors + static_cast<uint32_t>(needsTail);
    layout.slotBytes = (slotVectors == 0U ? 1U : slotVectors) * kVectorBytes;
    layout.inverseOffset = kOutputSlots * layout.slotBytes;
    layout.tailOffset = layout.inverseOffset + layout.slotBytes;
    // Each caller output credit also owns its padded tail input/FP8 output.
    layout.bytes = layout.tailOffset + (needsTail ? kOutputSlots * kTileElements * sizeof(uint16_t) : 0U);
    return layout;
}
} // namespace MegaMhc::QuantOutput

#undef MEGA_MHC_QUANT_LAYOUT_INLINE
#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_QUANT_LAYOUT_H
