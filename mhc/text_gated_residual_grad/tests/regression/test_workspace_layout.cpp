/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cassert>
#include <cstdint>
#include <iostream>
#include "../../op_kernel/arch35/text_gated_residual_grad_pipeline_arch35.h"

namespace {
constexpr uint32_t FEATURES_BELOW_UINT32_SLOT = 65280;
constexpr uint32_t FEATURES_AT_UINT32_SLOT = 65536;
constexpr uint32_t FEATURES_ABOVE_UINT32_SLOT = 65792;
constexpr uint32_t PROJECTION = 65520;
constexpr uint32_t CHANNELS = 16;
constexpr uint32_t AIV_COUNT = 32;
constexpr uint32_t WORKSPACE_SLOTS = 3;
constexpr uint64_t EXPECTED_G56_ROWS = 65536;
} // namespace

int main()
{
    using namespace TextGatedResidualGrad;
    // Layout-only test: no allocation. Cross UINT32_MAX without relying on today's UB/core limits.
    for (uint32_t d : {FEATURES_BELOW_UINT32_SLOT, FEATURES_AT_UINT32_SLOT, FEATURES_ABOVE_UINT32_SLOT}) {
        const uint64_t expectedSlot = EXPECTED_G56_ROWS * d;
        const auto layout = BuildHcGradWorkspaceLayout(WORKSPACE_SLOTS, 1, AIV_COUNT, CHANNELS, d / CHANNELS,
                                                       PROJECTION, d, sizeof(uint16_t), true);
        assert(layout.dwSlotElems == expectedSlot);
        assert(layout.dwGateRowElems == static_cast<uint64_t>(PROJECTION) * d);
        assert(layout.dactPartial - layout.dwPartial == HcGradAlign(WORKSPACE_SLOTS * expectedSlot * sizeof(float)));
        assert(layout.totalBytes > layout.dactPartial);
        const uint64_t projectionBytes = HcGradAlign(static_cast<uint64_t>(PROJECTION) * d * sizeof(float));
        assert(layout.dwUpAccum - layout.dwDownAccum == projectionBytes);
        assert(layout.dwGateAccum - layout.dwUpAccum == projectionBytes);
        assert(layout.totalBytes - layout.dwGateAccum ==
               HcGradAlign(static_cast<uint64_t>(CHANNELS) * d * sizeof(float)));
        const auto empty =
            BuildHcGradWorkspaceLayout(0, 1, AIV_COUNT, CHANNELS, d / CHANNELS, PROJECTION, d, sizeof(uint16_t), true);
        assert(empty.dwDownAccum == empty.totalBytes);
        assert(empty.dwUpAccum == empty.totalBytes);
        assert(empty.dwGateAccum == empty.totalBytes);
    }
    std::cout << "PASS: workspace slot sizes below, at and above 2^32 elements\n";
}
