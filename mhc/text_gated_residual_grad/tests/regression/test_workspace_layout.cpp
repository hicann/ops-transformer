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
constexpr uint32_t FULL_K_TEST_CHANNELS = 4;
constexpr uint32_t FULL_K_TEST_HIDDEN_SIZE = 2560;
constexpr uint32_t FULL_K_TEST_PROJECTION = 320;
constexpr uint32_t FULL_K_TEST_FEATURES = FULL_K_TEST_CHANNELS * FULL_K_TEST_HIDDEN_SIZE;
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
    // The bounded token-ring fallback must not change any workspace region with BS.
    uint64_t boundedBytes = 0;
    for (uint32_t n : {16384U, 32768U, 65536U}) {
        const uint32_t rows = ChooseHcGradTaskRows(n, 24U, 320U, 128U, 256U);
        const uint32_t tasks = (n + rows - 1U) / rows;
        const uint32_t slots = tasks < 3U ? tasks : 3U;
        const auto l =
            BuildHcGradWorkspaceLayout(slots, 1, 48U, 4, 2560, 320, 10240, sizeof(uint16_t), true, false, rows);
        assert(l.weightRows == 0U && !l.fullBatch);
        assert(l.dup - l.xn == HcGradAlign(static_cast<uint64_t>(slots) * rows * 10240U * sizeof(uint16_t)));
        assert(l.direct - l.dup == HcGradAlign(static_cast<uint64_t>(slots) * rows * 10240U * sizeof(uint16_t)));
        assert(l.packedT - l.packed == HcGradAlign(static_cast<uint64_t>(slots) * rows * l.g56Rows * sizeof(uint16_t)));
        assert(l.TaskBufferIndex(l.xn, slots, slots) == 0U);
        assert(l.TaskBufferIndex(l.act, slots, slots) == 0U);
        if (boundedBytes == 0U) {
            boundedBytes = l.totalBytes;
        } else {
            assert(l.totalBytes == boundedBytes);
        }
    }

    // A2 retains one fixed full-batch chunk rather than the complete BS.
    uint64_t chunkBytes = 0;
    for (uint32_t n : {16384U, 32768U, 65536U}) {
        const uint32_t taskRows = ChooseHcGradTaskRows(n, 20U, 320U, 128U, 256U);
        const uint32_t tasks = (n + taskRows - 1U) / taskRows;
        const uint32_t slots = tasks < 3U ? tasks : 3U;
        const auto l = BuildHcGradWorkspaceLayout(slots, 1, 40U, 4, 2560, 320, 10240, sizeof(uint16_t), true, false,
                                                  taskRows, HC_GRAD_CHUNK_ROWS, true, 324, true, true, true, true);
        assert(l.weightRows == HC_GRAD_CHUNK_ROWS && l.fullBatch && l.recomputeDirect && l.duplicateV1Operands);
        assert(l.dx2 == l.direct);
        assert(l.wConcat - l.dx2 == HcGradAlign(1ULL * HC_GRAD_CHUNK_ROWS * 10240U * sizeof(float)));
        assert(l.dx2 >= l.dup + HcGradAlign(2ULL * HC_GRAD_CHUNK_ROWS * 10240U * sizeof(uint16_t)));
        assert(l.dwDownAccum == l.dwPartial && l.dwUpAccum == l.dwUpPartial);
        assert(l.dwGateAccum == l.dwPartial + static_cast<uint64_t>(320U) * 10240U * sizeof(float));
        assert(l.dup - l.xn == HcGradAlign(2ULL * HC_GRAD_CHUNK_ROWS * 10240U * sizeof(uint16_t)));
        assert(l.direct - l.dup == HcGradAlign(2ULL * HC_GRAD_CHUNK_ROWS * 10240U * sizeof(uint16_t)));
        const uint64_t xnSlotBytes = static_cast<uint64_t>(taskRows) * 10240U * sizeof(uint16_t);
        assert(l.TaskBufferOffset(l.xn, 0U, slots, xnSlotBytes, 1U) ==
               l.xn + static_cast<uint64_t>(HC_GRAD_CHUNK_ROWS) * 10240U * sizeof(uint16_t));
        if (chunkBytes == 0U) {
            chunkBytes = l.totalBytes;
        } else {
            assert(l.totalBytes == chunkBytes);
        }
    }

    // The opt-in full-K layout still retains every input row even after the token ring wraps.
    for (uint32_t cores : {16U, 20U, 24U, 28U, 32U}) {
        for (uint32_t n : {1U, 17U, 1281U, 3841U, 16384U, 32768U, 49152U}) {
            const uint32_t rows = ChooseHcGradTaskRows(n, cores, FULL_K_TEST_PROJECTION, 128U, 256U);
            const uint32_t tasks = (n + rows - 1U) / rows;
            const uint32_t slots = tasks < 3U ? tasks : 3U;
            const auto l = BuildHcGradWorkspaceLayout(slots, 1, 2 * cores, FULL_K_TEST_CHANNELS,
                                                      FULL_K_TEST_HIDDEN_SIZE, FULL_K_TEST_PROJECTION,
                                                      FULL_K_TEST_FEATURES, sizeof(uint16_t), true, false, rows, n);
            struct Region {
                uint64_t start, end;
                uint32_t stride;
            };
            for (const auto& region :
                 {Region{l.xn, l.dup, FULL_K_TEST_FEATURES}, Region{l.dup, l.direct, FULL_K_TEST_FEATURES},
                  Region{l.act, l.dact, FULL_K_TEST_PROJECTION}, Region{l.packed, l.packedT, l.g56Rows}}) {
                uint64_t previousEnd = region.start;
                for (uint32_t task = 0; task < tasks; ++task) {
                    const uint32_t validRows = n - task * rows < rows ? n - task * rows : rows;
                    const uint64_t start =
                        region.start + static_cast<uint64_t>(l.TaskBufferIndex(region.start, task, slots)) * rows *
                                           region.stride * sizeof(uint16_t);
                    const uint64_t end = start + static_cast<uint64_t>(validRows) * region.stride * sizeof(uint16_t);
                    assert(start == previousEnd && end <= region.end);
                    previousEnd = end;
                }
                assert(region.end - region.start == HcGradAlign(static_cast<uint64_t>(n) * region.stride * 2U));
            }
            const uint64_t dxSlotBytes = static_cast<uint64_t>(rows) * FULL_K_TEST_FEATURES * sizeof(float);
            assert(l.dx2 - l.direct == HcGradAlign(slots * dxSlotBytes));
            assert(l.TaskBufferIndex(l.direct, slots, slots) == 0U);
            assert(l.dactPartial - l.dwPartial == HcGradAlign(l.dwSlotElems * sizeof(float)));
            const bool separateTail = tasks > slots && n % rows != 0U;
            if (separateTail) {
                const uint64_t paddedTail = HcGradAlign16(n % rows);
                assert(l.cubeTailTask == tasks - 1U);
                const uint64_t dwGateBytes =
                    static_cast<uint64_t>(FULL_K_TEST_CHANNELS) * FULL_K_TEST_FEATURES * sizeof(float);
                const uint64_t dactSlotBytes = static_cast<uint64_t>(rows) * FULL_K_TEST_PROJECTION * sizeof(float);
                assert(l.cubeTailDact >= l.dwGateAccum + HcGradAlign(dwGateBytes));
                assert(l.cubeTailDx - l.cubeTailDact ==
                       HcGradAlign(paddedTail * FULL_K_TEST_PROJECTION * sizeof(float)));
                assert(l.totalBytes - l.cubeTailDx == HcGradAlign(paddedTail * FULL_K_TEST_FEATURES * sizeof(float)));
                assert(l.TaskBufferOffset(l.dx2, tasks - 1, slots, dxSlotBytes) == l.cubeTailDx);
                assert(l.TaskBufferOffset(l.dactPartial, tasks - 1, slots, dactSlotBytes) == l.cubeTailDact);
            } else {
                assert(l.cubeTailTask == 0xffffffffU);
                assert(l.cubeTailDx == 0U && l.cubeTailDact == 0U);
            }
            for (uint32_t task = 0; task + 1 < tasks; ++task) {
                assert(l.TaskBufferOffset(l.dx2, task, slots, dxSlotBytes) == l.dx2 + (task % slots) * dxSlotBytes);
            }
        }
    }
    std::cout << "PASS: workspace overflow boundaries, fixed A2 chunk and BS-invariant bounded ring\n";
}
