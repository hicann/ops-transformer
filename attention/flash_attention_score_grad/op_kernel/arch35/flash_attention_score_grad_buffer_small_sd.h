/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_buffer_small_sd.h
 * \brief Small-S/Small-D dedicated const, slot, and static buffer layout definitions.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BUFFER_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_BUFFER_SMALL_SD_H

#include <cstddef>
#include <cstdint>

namespace FagBaseApi {

enum class SmallSDSlotState : uint8_t {
    EMPTY = 0,
    PREPARED,
    CUBE_INFLIGHT,
    READY_FOR_VECTOR,
    VECTOR_INFLIGHT,
    REUSABLE
};

struct SmallSDConstInfo {
    uint32_t b = 0;
    uint32_t n1 = 0;
    uint32_t n2 = 0;
    uint32_t g = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t d = 0;
    uint32_t dv = 0;
    uint32_t layout = 0;
    uint32_t inputDtype = 0;
    uint32_t outputDtype = 0;
    uint32_t calcTypeSize = sizeof(float);
    uint32_t usedCoreNum = 0;
    uint32_t taskCount = 0;
    uint32_t isTnd = 0;
    uint32_t isSingleTask = 0;
    uint32_t s2Align16 = 0;
    float scale = 1.0f;

    uint64_t qStrideB = 0;
    uint64_t qStrideN2 = 0;
    uint64_t kStrideB = 0;
    uint64_t kStrideN2 = 0;
    uint64_t vStrideB = 0;
    uint64_t vStrideN2 = 0;
    uint64_t dyStrideB = 0;
    uint64_t dyStrideN2 = 0;
    uint64_t dqStrideB = 0;
    uint64_t dqStrideN2 = 0;
    uint64_t dkStrideB = 0;
    uint64_t dkStrideN2 = 0;
    uint64_t dvStrideB = 0;
    uint64_t dvStrideN2 = 0;

    uint64_t qMatrixBytes = 0;
    uint64_t kMatrixBytes = 0;
    uint64_t vMatrixBytes = 0;
    uint64_t dyMatrixBytes = 0;
    uint64_t dqMatrixBytes = 0;
    uint64_t dkMatrixBytes = 0;
    uint64_t dvMatrixBytes = 0;
    uint64_t cubeResultBytes = 0;
    uint64_t vectorTempBytes = 0;

    uint64_t workspaceBaseOffset = 0;
    uint64_t workspaceSize = 0;
};

struct SmallSDTaskCursor {
    int64_t bIdx = 0;
    int64_t n2oIdx = 0;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t s2SizeAcc = 0;
    int64_t b1SSOffset = 0;
    int64_t b1SSOffsetAlign = 0;
    int64_t lastBatchTotalBaseIdx = 0;
    int64_t lastBatchTotalS1BOffset = 0;
    int64_t lastBatchTotalS2BOffset = 0;
    int64_t lastBatchTotalS1BOffsetForDv = 0;
    int64_t lastBatchTotalS2BOffsetForDv = 0;
    int64_t lastBatchTotalS1S2SizeAlign = 0;
    int64_t lastBatchTotalS1S2Size = 0;
    int64_t lastBatchTotalS2Size = 0;
    int64_t qOffset = 0;
    int64_t kOffset = 0;
    int64_t qTaskStride = 0;
    int64_t kTaskStride = 0;
    int64_t qBatchGap = 0;
    int64_t kBatchGap = 0;
};

struct SmallSDShape {
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t s2Align16 = 0;
    int64_t halfS1 = 0;
    int64_t firstHalfS1 = 0;
    int64_t halfS2 = 0;
    int64_t firstHalfS2 = 0;
};

struct SmallSDOffsets {
    int64_t q = 0;
    int64_t k = 0;
    int64_t attention = 0;
    int64_t attentionAlign = 0;
    int64_t s2Prefix = 0;
};

struct SmallSDPipelineSlot {
    int64_t taskId = -1;
    int64_t taskIdMod2 = 0;
    int64_t bIdx = 0;
    int64_t n2oIdx = 0;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t s2AlignedSize = 0;
    int64_t halfS1 = 0;
    int64_t firstHalfS1 = 0;
    int64_t halfS2 = 0;
    int64_t firstHalfS2 = 0;
    int64_t vecCoreOffset = 0;
    int64_t qOffset = 0;
    int64_t kOffset = 0;
    int64_t attentionOffset = 0;
    int64_t attentionAlignOffset = 0;
    int64_t s2Prefix = 0;
    int64_t lastBatchTotalBaseIdx = 0;
    int64_t lastBatchTotalS1BOffset = 0;
    int64_t lastBatchTotalS2BOffset = 0;
    int64_t lastBatchTotalS1BOffsetForDv = 0;
    int64_t lastBatchTotalS2BOffsetForDv = 0;
    int64_t lastBatchTotalS1S2SizeAlign = 0;
    int64_t lastBatchTotalS1S2Size = 0;
    int64_t lastBatchTotalS2Size = 0;
    SmallSDSlotState state = SmallSDSlotState::EMPTY;
};

static_assert(sizeof(SmallSDConstInfo) == 272, "SmallSDConstInfo ABI size changed unexpectedly.");
static_assert(offsetof(SmallSDConstInfo, qStrideB) == 72, "SmallSDConstInfo stride offset changed unexpectedly.");
static_assert(offsetof(SmallSDConstInfo, workspaceBaseOffset) == 256,
              "SmallSDConstInfo workspace offset changed unexpectedly.");
static_assert(sizeof(SmallSDTaskCursor) == 168, "SmallSDTaskCursor ABI size changed unexpectedly.");
static_assert(offsetof(SmallSDTaskCursor, qOffset) == 120, "SmallSDTaskCursor qOffset changed unexpectedly.");
static_assert(sizeof(SmallSDShape) == 56, "SmallSDShape ABI size changed unexpectedly.");
static_assert(sizeof(SmallSDOffsets) == 40, "SmallSDOffsets ABI size changed unexpectedly.");
static_assert(sizeof(SmallSDPipelineSlot) == 208, "SmallSDPipelineSlot ABI size changed unexpectedly.");
static_assert(offsetof(SmallSDPipelineSlot, state) == 200, "SmallSDPipelineSlot state offset changed unexpectedly.");

template <uint32_t HEAD_DIM>
struct SmallSDBufferLayout {
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "SmallSD buffer layout supports D=64 or D=128 only.");
    static constexpr uint32_t slotCount = 2;
    static constexpr uint32_t maxS1 = 128;
    static constexpr uint32_t maxS2 = 128;
    static constexpr uint32_t alignBytes = 32;
    static constexpr uint32_t inputElemBytes = 2;
    static constexpr uint32_t calcElemBytes = 4;

    static constexpr uint32_t qTileBytes = maxS1 * HEAD_DIM * inputElemBytes;
    static constexpr uint32_t kTileBytes = maxS2 * HEAD_DIM * inputElemBytes;
    static constexpr uint32_t vTileBytes = maxS2 * HEAD_DIM * inputElemBytes;
    static constexpr uint32_t dyTileBytes = maxS1 * HEAD_DIM * inputElemBytes;
    static constexpr uint32_t qkResultBytes = maxS1 * maxS2 * calcElemBytes;
    static constexpr uint32_t dyvResultBytes = maxS1 * maxS2 * calcElemBytes;
    static constexpr uint32_t dsL1Bytes = maxS1 * maxS2 * inputElemBytes;
    static constexpr uint32_t pL1Bytes = maxS1 * maxS2 * inputElemBytes;
    static constexpr uint32_t dqUbBytes = maxS1 * HEAD_DIM * calcElemBytes;
    static constexpr uint32_t dkUbBytes = maxS2 * HEAD_DIM * calcElemBytes;
    static constexpr uint32_t dvUbBytes = maxS2 * HEAD_DIM * calcElemBytes;
    static constexpr uint32_t vectorTempBytes = maxS1 * maxS2 * calcElemBytes;

    static constexpr uint32_t perSlotBytes =
        qkResultBytes + dyvResultBytes + dqUbBytes + dkUbBytes + dvUbBytes + vectorTempBytes;
    static constexpr uint32_t sharedReadonlyBytes = qTileBytes + kTileBytes + vTileBytes + dyTileBytes;
    static constexpr uint32_t sharedL1Bytes = dsL1Bytes + pL1Bytes;
    static constexpr uint32_t slot0Offset = 0;
    static constexpr uint32_t slot1Offset = slot0Offset + perSlotBytes;
    static constexpr uint32_t sharedReadonlyOffset = slot1Offset + perSlotBytes;
    static constexpr uint32_t sharedL1Offset = sharedReadonlyOffset + sharedReadonlyBytes;
    static constexpr uint32_t totalBytes = sharedL1Offset + sharedL1Bytes;

    static constexpr uint32_t QkResultOffset(uint32_t slotIdx)
    {
        return (slotIdx == 0 ? slot0Offset : slot1Offset);
    }
    static constexpr uint32_t DyVResultOffset(uint32_t slotIdx)
    {
        return QkResultOffset(slotIdx) + qkResultBytes;
    }
    static constexpr uint32_t DqUbOffset(uint32_t slotIdx)
    {
        return DyVResultOffset(slotIdx) + dyvResultBytes;
    }
    static constexpr uint32_t DkUbOffset(uint32_t slotIdx)
    {
        return DqUbOffset(slotIdx) + dqUbBytes;
    }
    static constexpr uint32_t DvUbOffset(uint32_t slotIdx)
    {
        return DkUbOffset(slotIdx) + dkUbBytes;
    }
    static constexpr uint32_t VectorTempOffset(uint32_t slotIdx)
    {
        return DvUbOffset(slotIdx) + dvUbBytes;
    }
};

static_assert(SmallSDBufferLayout<64>::slot1Offset == SmallSDBufferLayout<64>::perSlotBytes,
              "SmallSD D=64 slot offsets must be deterministic.");
static_assert(SmallSDBufferLayout<128>::slot1Offset == SmallSDBufferLayout<128>::perSlotBytes,
              "SmallSD D=128 slot offsets must be deterministic.");
static_assert((SmallSDBufferLayout<64>::totalBytes % SmallSDBufferLayout<64>::alignBytes) == 0,
              "SmallSD D=64 buffer layout must be 32-byte aligned.");
static_assert((SmallSDBufferLayout<128>::totalBytes % SmallSDBufferLayout<128>::alignBytes) == 0,
              "SmallSD D=128 buffer layout must be 32-byte aligned.");

} // namespace FagBaseApi

#endif
