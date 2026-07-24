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
    uint32_t tndMaxSumLayout = 0;
    float scale = 1.0f;

    uint64_t qStrideB = 0;
    uint64_t qStrideN2 = 0;
    uint64_t qStrideS = 0;
    uint64_t kStrideB = 0;
    uint64_t kStrideN2 = 0;
    uint64_t kStrideS = 0;
    uint64_t vStrideB = 0;
    uint64_t vStrideN2 = 0;
    uint64_t vStrideS = 0;
    uint64_t dyStrideB = 0;
    uint64_t dyStrideN2 = 0;
    uint64_t dyStrideS = 0;
    uint64_t dqStrideB = 0;
    uint64_t dqStrideN2 = 0;
    uint64_t dqStrideS = 0;
    uint64_t dkStrideB = 0;
    uint64_t dkStrideN2 = 0;
    uint64_t dkStrideS = 0;
    uint64_t dvStrideB = 0;
    uint64_t dvStrideN2 = 0;
    uint64_t dvStrideS = 0;
    uint64_t attentionStrideB = 0;
    uint64_t attentionStrideN2 = 0;
    uint64_t attentionStrideS = 0;
    uint64_t softmaxStrideB = 0;
    uint64_t softmaxStrideN2 = 0;
    uint64_t softmaxStrideS = 0;

    uint64_t qMatrixElements = 0;
    uint64_t kMatrixElements = 0;
    uint64_t vMatrixElements = 0;
    uint64_t dyMatrixElements = 0;
    uint64_t dqMatrixElements = 0;
    uint64_t dkMatrixElements = 0;
    uint64_t dvMatrixElements = 0;
    uint64_t cubeResultElements = 0;
    uint64_t vectorTempElements = 0;
    uint64_t qMatrixBytes = 0;
    uint64_t kMatrixBytes = 0;
    uint64_t vMatrixBytes = 0;
    uint64_t dyMatrixBytes = 0;
    uint64_t dqMatrixBytes = 0;
    uint64_t dkMatrixBytes = 0;
    uint64_t dvMatrixBytes = 0;
    uint64_t cubeResultBytes = 0;
    uint64_t vectorTempBytes = 0;
    uint32_t dTemplateCapacity = 0;
    uint32_t aivHalfS1 = 0;
    uint32_t aivFirstHalfS1 = 0;
    uint32_t aivHalfS2 = 0;
    uint32_t aivFirstHalfS2 = 0;
    uint32_t reserved = 0;

    uint64_t workspaceBaseOffset = 0;
    uint64_t workspaceSize = 0;
};

struct SmallSDTaskCursor {
    int64_t bIdx = 0;
    int64_t n2oIdx = 0;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t softmaxPrefix = 0;
    int64_t attentionPrefix = 0;
    int64_t alignedAttentionPrefix = 0;
    int64_t baseTaskPrefix = 0;
    int64_t qPrefix = 0;
    int64_t kvPrefix = 0;
    int64_t qDyDqPrefix = 0;
    int64_t kvDkDvPrefix = 0;
    int64_t taskRemaining = 0;
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
    int64_t vOffset = 0;
    int64_t dyOffset = 0;
    int64_t dqOffset = 0;
    int64_t dkOffset = 0;
    int64_t dvOffset = 0;
    int64_t attentionOffset = 0;
    int64_t attentionAlignOffset = 0;
    int64_t s2Prefix = 0;
    int64_t baseTaskPrefix = 0;
    int64_t qPrefix = 0;
    int64_t kvPrefix = 0;
    int64_t qDyDqPrefix = 0;
    int64_t kvDkDvPrefix = 0;
};

static_assert(sizeof(SmallSDConstInfo) == 480, "SmallSDConstInfo ABI size changed unexpectedly.");
static_assert(offsetof(SmallSDConstInfo, qStrideB) == 80, "SmallSDConstInfo stride offset changed unexpectedly.");
static_assert(offsetof(SmallSDConstInfo, workspaceBaseOffset) == 464,
              "SmallSDConstInfo workspace offset changed unexpectedly.");
static_assert(sizeof(SmallSDTaskCursor) == 152, "SmallSDTaskCursor ABI size changed unexpectedly.");
static_assert(offsetof(SmallSDTaskCursor, qOffset) == 104, "SmallSDTaskCursor qOffset changed unexpectedly.");
static_assert(sizeof(SmallSDShape) == 56, "SmallSDShape ABI size changed unexpectedly.");
static_assert(sizeof(SmallSDOffsets) == 40, "SmallSDOffsets ABI size changed unexpectedly.");
static_assert(sizeof(SmallSDPipelineSlot) == 216, "SmallSDPipelineSlot ABI size changed unexpectedly.");

template <uint32_t HEAD_DIM>
struct SmallSDBufferLayout {
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "SmallSD buffer layout supports D=64 or D=128 only.");
    static constexpr uint32_t slotCount = 2;
    static constexpr uint32_t maxS1 = 128;
    static constexpr uint32_t maxS2 = 128;
    static constexpr uint32_t alignBytes = 32;
    static constexpr uint32_t inputElemBytes = 2;
    static constexpr uint32_t calcElemBytes = 4;
    static constexpr uint32_t l1CapacityBytes = 1024 * 1024;
    static constexpr uint32_t l0aCapacityBytes = 64 * 1024;
    static constexpr uint32_t l0bCapacityBytes = 64 * 1024;
    static constexpr uint32_t l0cCapacityBytes = 256 * 1024;
    static constexpr uint32_t ubCapacityBytes = 512 * 1024;

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
    static constexpr uint32_t l0PingBytes = 32 * 1024;
    static constexpr uint32_t l0PongBytes = 32 * 1024;

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
    static constexpr uint32_t QOffset()
    {
        return sharedReadonlyOffset;
    }
    static constexpr uint32_t KOffset()
    {
        return QOffset() + qTileBytes;
    }
    static constexpr uint32_t VOffset()
    {
        return KOffset() + kTileBytes;
    }
    static constexpr uint32_t DyOffset()
    {
        return VOffset() + vTileBytes;
    }
    static constexpr uint32_t DsOffset()
    {
        return sharedL1Offset;
    }
    static constexpr uint32_t POffset()
    {
        return DsOffset() + dsL1Bytes;
    }
    static constexpr uint32_t Mm1ResultOffset(uint32_t slotIdx)
    {
        return QkResultOffset(slotIdx);
    }
    static constexpr uint32_t Mm2ResultOffset(uint32_t slotIdx)
    {
        return DyVResultOffset(slotIdx);
    }
    static constexpr uint32_t L0APingOffset()
    {
        return 0;
    }
    static constexpr uint32_t L0APongOffset()
    {
        return L0APingOffset() + l0PingBytes;
    }
    static constexpr uint32_t L0BPingOffset()
    {
        return 0;
    }
    static constexpr uint32_t L0BPongOffset()
    {
        return L0BPingOffset() + l0PingBytes;
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
static_assert(SmallSDBufferLayout<64>::sharedL1Bytes + SmallSDBufferLayout<64>::sharedReadonlyBytes <=
                  SmallSDBufferLayout<64>::l1CapacityBytes,
              "SmallSD D=64 L1 layout exceeds capacity.");
static_assert(SmallSDBufferLayout<128>::sharedL1Bytes + SmallSDBufferLayout<128>::sharedReadonlyBytes <=
                  SmallSDBufferLayout<128>::l1CapacityBytes,
              "SmallSD D=128 L1 layout exceeds capacity.");
static_assert(SmallSDBufferLayout<64>::l0PingBytes + SmallSDBufferLayout<64>::l0PongBytes <=
                  SmallSDBufferLayout<64>::l0aCapacityBytes,
              "SmallSD D=64 L0A ping/pong exceeds capacity.");
static_assert(SmallSDBufferLayout<128>::l0PingBytes + SmallSDBufferLayout<128>::l0PongBytes <=
                  SmallSDBufferLayout<128>::l0aCapacityBytes,
              "SmallSD D=128 L0A ping/pong exceeds capacity.");
static_assert(SmallSDBufferLayout<64>::l0PingBytes + SmallSDBufferLayout<64>::l0PongBytes <=
                  SmallSDBufferLayout<64>::l0bCapacityBytes,
              "SmallSD D=64 L0B ping/pong exceeds capacity.");
static_assert(SmallSDBufferLayout<128>::l0PingBytes + SmallSDBufferLayout<128>::l0PongBytes <=
                  SmallSDBufferLayout<128>::l0bCapacityBytes,
              "SmallSD D=128 L0B ping/pong exceeds capacity.");
static_assert(SmallSDBufferLayout<64>::qkResultBytes <= SmallSDBufferLayout<64>::l0cCapacityBytes,
              "SmallSD D=64 L0C score buffer exceeds capacity.");
static_assert(SmallSDBufferLayout<128>::qkResultBytes <= SmallSDBufferLayout<128>::l0cCapacityBytes,
              "SmallSD D=128 L0C score buffer exceeds capacity.");
static_assert(SmallSDBufferLayout<64>::perSlotBytes <= SmallSDBufferLayout<64>::ubCapacityBytes,
              "SmallSD D=64 UB slot buffer exceeds capacity.");
static_assert(SmallSDBufferLayout<128>::perSlotBytes <= SmallSDBufferLayout<128>::ubCapacityBytes,
              "SmallSD D=128 UB slot buffer exceeds capacity.");

} // namespace FagBaseApi

#endif
