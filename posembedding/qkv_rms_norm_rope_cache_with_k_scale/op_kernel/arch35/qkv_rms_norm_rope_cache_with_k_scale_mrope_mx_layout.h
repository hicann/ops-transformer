/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_LAYOUT_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_LAYOUT_H_

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace QkvRmsNormRopeCacheWithKScale {

#define QKV_MROPE_MX_HOST_DEVICE

constexpr uint64_t MROPE_MX_UB_ALIGN_BYTES = 32U;
constexpr uint64_t MROPE_MX_POSITION_WINDOW_TOKENS = 1024U;
constexpr uint64_t MROPE_MX_HEAD_DIM_D128 = 128U;
constexpr uint64_t MROPE_MX_BF16_BYTES = 2U;
constexpr uint64_t MROPE_MX_FP32_BYTES = 4U;
constexpr uint64_t MROPE_MX_INT32_BYTES = 4U;
constexpr uint64_t MROPE_MX_FP8_BYTES = 1U;
constexpr uint64_t MROPE_MX_SCALE_COUNT_D128 = 4U;
constexpr uint64_t MROPE_MX_ROW_BATCH_ROWS = 16U;
constexpr uint64_t MROPE_MX_ROW_BATCH_DATA_BYTES =
    MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_HEAD_DIM_D128 * MROPE_MX_FP32_BYTES;
constexpr uint64_t MROPE_MX_ROW_BATCH_MAX_BYTES =
    MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128 * MROPE_MX_FP32_BYTES;
constexpr uint64_t MROPE_MX_ROW_BATCH_RECIP_BYTES =
    MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128 * MROPE_MX_FP32_BYTES;
// One aligned FP32 register per head pair.  Only lanes 0 and 1 carry the two
// row-wise mean-square values; the padding keeps every pair store 32B aligned.
constexpr uint64_t MROPE_MX_ROW_BATCH_RMS_SQUARED_BYTES = (MROPE_MX_ROW_BATCH_ROWS / 2U) * MROPE_MX_UB_ALIGN_BYTES;
constexpr uint64_t MROPE_MX_ROW_BATCH_SCRATCH_BYTES = MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES +
                                                      MROPE_MX_ROW_BATCH_RECIP_BYTES +
                                                      MROPE_MX_ROW_BATCH_RMS_SQUARED_BYTES;
constexpr uint64_t MROPE_MX_UINT32_MAX = 0xffffffffULL;
constexpr uint64_t MROPE_MX_UINT64_MAX = 0xffffffffffffffffULL;

/*
 * Current per-AIV UB allocation, in increasing byte-offset order:
 *
 *   persistent parameters
 *     qGamma | kGamma | vScale | M-RoPE gather index | position window
 *   double-buffered tile inputs
 *     slot 0: qkv | rawCosSin | slotMapping
 *     slot 1: qkv | rawCosSin | slotMapping
 *   single-buffered Vector-to-MTE3 staging
 *     V data | K data | K scale | Q data | Q scale
 *   Vector-only MX scratch
 *     RowBatch16 record 0 | ... | RowBatch16 record R-1
 *
 * Every region starts on a 32-byte boundary and regions do not alias. The
 * persistent regions occupy one physical allocation for the AIV token range;
 * only position contents are refreshed in 1024-token windows. Input slots are
 * alternated by tile ordinal and remain owned until Q, the last input consumer,
 * returns the slot. V/K/Q staging regions are distinct single buffers whose
 * reuse is protected by their MTE3_V FREE events. MX scratch is private to the
 * active VF call and its internal reuse is ordered by LocalMemBar.
 *
 * One RowBatch16 record contains, in order, 16xD128 FP32 staged data,
 * 16x4 FP32/uint32 maxima, 16x4 FP32 combined reciprocals, and eight aligned
 * head-pair rms-squared records. The base layout reserves
 * max(1, floor(Nq/16)) records. The Q GlobalTileWave layout expands this to
 * floor(Nq/16)*tokenTile records plus ceil(tokenTile/2) records when Nq has an
 * eight-head tail. Host capacity checking and device address materialization
 * both use the allocation schema in this header; changing a region size or
 * order must update both builders together.
 */
struct MropeMxUbLayout {
    uint32_t qGammaOffsetBytes;
    uint32_t kGammaOffsetBytes;
    uint32_t vScaleOffsetBytes;
    uint32_t gatherIndexOffsetBytes;
    uint32_t positionOffsetBytes;

    uint32_t inputSlotOffsetBytes[2];
    uint32_t inputQkvOffsetBytes[2];
    uint32_t inputRawCosSinOffsetBytes[2];
    uint32_t inputSlotMappingOffsetBytes[2];
    uint32_t inputSlotBytes;

    uint32_t vDataOffsetBytes;
    uint32_t vDataBytes;
    uint32_t kDataOffsetBytes;
    uint32_t kScaleOffsetBytes;
    uint32_t kSlotBytes;
    uint32_t qDataOffsetBytes;
    uint32_t qScaleOffsetBytes;
    uint32_t qSlotBytes;

    uint32_t mxScratchOffsetBytes;
    uint32_t mxScratchBytes;

    uint32_t totalBytes;
};

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxCheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs > MROPE_MX_UINT64_MAX - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxCheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0U && rhs > MROPE_MX_UINT64_MAX / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxCheckedAlign(uint64_t value, uint64_t &result)
{
    uint64_t withPadding = 0U;
    if (!MropeMxCheckedAdd(value, MROPE_MX_UB_ALIGN_BYTES - 1U, withPadding)) {
        return false;
    }
    result = withPadding / MROPE_MX_UB_ALIGN_BYTES * MROPE_MX_UB_ALIGN_BYTES;
    return true;
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxNarrow(uint64_t value, uint32_t &result)
{
    if (value > MROPE_MX_UINT32_MAX) {
        return false;
    }
    result = static_cast<uint32_t>(value);
    return true;
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxAppendRegion(uint64_t regionBytes, uint64_t &cursor, uint32_t &offset,
                                                            uint32_t *size = nullptr)
{
    uint64_t alignedCursor = 0U;
    uint64_t alignedBytes = 0U;
    uint64_t next = 0U;
    if (!MropeMxCheckedAlign(cursor, alignedCursor) || !MropeMxCheckedAlign(regionBytes, alignedBytes) ||
        !MropeMxCheckedAdd(alignedCursor, alignedBytes, next) || !MropeMxNarrow(alignedCursor, offset)) {
        return false;
    }
    if (size != nullptr && !MropeMxNarrow(alignedBytes, *size)) {
        return false;
    }
    cursor = next;
    return true;
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxProduct2(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    return MropeMxCheckedMul(lhs, rhs, result);
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxProduct3(uint64_t lhs, uint64_t mid, uint64_t rhs, uint64_t &result)
{
    uint64_t partial = 0U;
    return MropeMxCheckedMul(lhs, mid, partial) && MropeMxCheckedMul(partial, rhs, result);
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool MropeMxProduct4(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                                                        uint64_t &result)
{
    uint64_t partial = 0U;
    return MropeMxProduct3(a, b, c, partial) && MropeMxCheckedMul(partial, d, result);
}

QKV_MROPE_MX_HOST_DEVICE constexpr bool TryMakeMropeMxUbLayout(uint64_t tokenTile, uint64_t qHeadNum,
                                                               uint64_t kvHeadNum, uint64_t headDim,
                                                               MropeMxUbLayout &layout)
{
    if (tokenTile == 0U || qHeadNum == 0U || kvHeadNum == 0U || headDim != MROPE_MX_HEAD_DIM_D128) {
        return false;
    }

    MropeMxUbLayout candidate{};
    uint64_t cursor = 0U;
    uint64_t bytes = 0U;
    uint64_t headCount = 0U;
    if (!MropeMxCheckedAdd(qHeadNum, kvHeadNum, headCount) || !MropeMxCheckedAdd(headCount, kvHeadNum, headCount)) {
        return false;
    }

    if (!MropeMxProduct2(headDim, MROPE_MX_FP32_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.qGammaOffsetBytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.kGammaOffsetBytes) ||
        !MropeMxProduct3(kvHeadNum, headDim, MROPE_MX_FP32_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.vScaleOffsetBytes) ||
        !MropeMxProduct2(headDim / 2U, MROPE_MX_INT32_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.gatherIndexOffsetBytes) ||
        !MropeMxProduct3(MROPE_MX_POSITION_WINDOW_TOKENS, 3U, MROPE_MX_INT32_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.positionOffsetBytes)) {
        return false;
    }

    for (uint32_t slot = 0U; slot < 2U; ++slot) {
        uint64_t slotBegin = 0U;
        if (!MropeMxCheckedAlign(cursor, slotBegin) ||
            !MropeMxNarrow(slotBegin, candidate.inputSlotOffsetBytes[slot]) ||
            !MropeMxProduct4(tokenTile, headCount, headDim, MROPE_MX_BF16_BYTES, bytes) ||
            !MropeMxAppendRegion(bytes, cursor, candidate.inputQkvOffsetBytes[slot]) ||
            !MropeMxProduct4(tokenTile, 3U, headDim, MROPE_MX_FP32_BYTES, bytes) ||
            !MropeMxAppendRegion(bytes, cursor, candidate.inputRawCosSinOffsetBytes[slot]) ||
            !MropeMxProduct2(tokenTile, MROPE_MX_INT32_BYTES, bytes) ||
            !MropeMxAppendRegion(bytes, cursor, candidate.inputSlotMappingOffsetBytes[slot])) {
            return false;
        }
        uint64_t slotEnd = 0U;
        if (!MropeMxCheckedAlign(cursor, slotEnd)) {
            return false;
        }
        if (slot == 0U && !MropeMxNarrow(slotEnd - slotBegin, candidate.inputSlotBytes)) {
            return false;
        }
        cursor = slotEnd;
    }

    uint64_t kScaleTokenBytes = 0U;
    if (!MropeMxProduct3(kvHeadNum, MROPE_MX_SCALE_COUNT_D128, MROPE_MX_FP8_BYTES, bytes) ||
        !MropeMxCheckedAlign(bytes, kScaleTokenBytes) ||
        !MropeMxProduct4(tokenTile, kvHeadNum, headDim, MROPE_MX_FP8_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.vDataOffsetBytes, &candidate.vDataBytes) ||
        !MropeMxProduct4(tokenTile, kvHeadNum, headDim, MROPE_MX_FP8_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.kDataOffsetBytes) ||
        !MropeMxProduct2(tokenTile, kScaleTokenBytes, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.kScaleOffsetBytes)) {
        return false;
    }
    uint64_t kSlotEnd = 0U;
    if (!MropeMxCheckedAlign(cursor, kSlotEnd) ||
        !MropeMxNarrow(kSlotEnd - candidate.kDataOffsetBytes, candidate.kSlotBytes)) {
        return false;
    }
    cursor = kSlotEnd;

    if (!MropeMxProduct4(tokenTile, qHeadNum, headDim, MROPE_MX_FP8_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.qDataOffsetBytes) ||
        !MropeMxProduct3(tokenTile, qHeadNum, MROPE_MX_SCALE_COUNT_D128, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.qScaleOffsetBytes)) {
        return false;
    }
    uint64_t qSlotEnd = 0U;
    if (!MropeMxCheckedAlign(cursor, qSlotEnd) ||
        !MropeMxNarrow(qSlotEnd - candidate.qDataOffsetBytes, candidate.qSlotBytes)) {
        return false;
    }
    cursor = qSlotEnd;

    const uint64_t fullBatchCount = qHeadNum / MROPE_MX_ROW_BATCH_ROWS;
    const uint64_t scratchBatchCount = fullBatchCount == 0U ? 1U : fullBatchCount;
    if (!MropeMxProduct2(scratchBatchCount, MROPE_MX_ROW_BATCH_SCRATCH_BYTES, bytes) ||
        !MropeMxAppendRegion(bytes, cursor, candidate.mxScratchOffsetBytes, &candidate.mxScratchBytes)) {
        return false;
    }
    if (!MropeMxNarrow(cursor, candidate.totalBytes)) {
        return false;
    }

    layout = candidate;
    return true;
}

// The unified Q schedule keeps every complete RowBatch16 record in the current
// token wave alive until the tile-wide Scale and Quant phases consume it.  An
// optional tail8 record is shared by two tokens; an odd final token occupies
// only its low eight rows.  No state crosses a tile or an AIV range.
QKV_MROPE_MX_HOST_DEVICE constexpr bool TryMakeMropeMxQGlobalTileWaveUbLayout(uint64_t tokenTile, uint64_t qHeadNum,
                                                                              uint64_t kvHeadNum, uint64_t headDim,
                                                                              MropeMxUbLayout &layout)
{
    MropeMxUbLayout candidate{};
    if (!TryMakeMropeMxUbLayout(tokenTile, qHeadNum, kvHeadNum, headDim, candidate)) {
        return false;
    }
    const uint64_t fullBatchCount = qHeadNum / MROPE_MX_ROW_BATCH_ROWS;
    const uint64_t currentScratchBatchCount = fullBatchCount == 0U ? 1U : fullBatchCount;
    const bool hasQTail = (qHeadNum % MROPE_MX_ROW_BATCH_ROWS) == MROPE_MX_ROW_BATCH_ROWS / 2U;
    uint64_t requiredScratchBatchCount = 0U;
    if (!MropeMxCheckedMul(fullBatchCount, tokenTile, requiredScratchBatchCount)) {
        return false;
    }
    if (hasQTail) {
        const uint64_t tailWaveBatchCount = (tokenTile + 1U) / 2U;
        if (!MropeMxCheckedAdd(requiredScratchBatchCount, tailWaveBatchCount, requiredScratchBatchCount)) {
            return false;
        }
    }
    if (requiredScratchBatchCount < currentScratchBatchCount) {
        requiredScratchBatchCount = currentScratchBatchCount;
    }
    const uint64_t additionalScratchBatchCount = requiredScratchBatchCount - currentScratchBatchCount;
    uint64_t additionalScratchBytes = 0U;
    uint64_t scratchBytes = 0U;
    uint64_t totalBytes = 0U;
    if (!MropeMxCheckedMul(additionalScratchBatchCount, MROPE_MX_ROW_BATCH_SCRATCH_BYTES, additionalScratchBytes) ||
        !MropeMxCheckedAdd(static_cast<uint64_t>(candidate.mxScratchBytes), additionalScratchBytes, scratchBytes) ||
        !MropeMxCheckedAdd(static_cast<uint64_t>(candidate.totalBytes), additionalScratchBytes, totalBytes)) {
        return false;
    }
    if (!MropeMxNarrow(scratchBytes, candidate.mxScratchBytes) || !MropeMxNarrow(totalBytes, candidate.totalBytes)) {
        return false;
    }
    layout = candidate;
    return true;
}

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t MropeMxAlignDevice(uint32_t value)
{
    return (value + static_cast<uint32_t>(MROPE_MX_UB_ALIGN_BYTES) - 1U) /
           static_cast<uint32_t>(MROPE_MX_UB_ALIGN_BYTES) * static_cast<uint32_t>(MROPE_MX_UB_ALIGN_BYTES);
}

__aicore__ inline void MropeMxAppendRegionDevice(uint32_t regionBytes, uint32_t &cursor, uint32_t &offset,
                                                 uint32_t *size = nullptr)
{
    cursor = MropeMxAlignDevice(cursor);
    offset = cursor;
    const uint32_t alignedBytes = MropeMxAlignDevice(regionBytes);
    if (size != nullptr) {
        *size = alignedBytes;
    }
    cursor += alignedBytes;
}

// Host tiling validates the same shape with TryMakeMropeMxUbLayout before
// launch. Device materialization intentionally uses only uint32 arithmetic so
// it cannot introduce compiler-rt wide-multiply dependencies.
__aicore__ inline void MakeMropeMxUbLayoutDevice(uint32_t tokenTile, uint32_t qHeadNum, uint32_t kvHeadNum,
                                                 uint32_t headDim, MropeMxUbLayout &layout)
{
    MropeMxUbLayout candidate{};
    uint32_t cursor = 0U;
    const uint32_t headCount = qHeadNum + kvHeadNum + kvHeadNum;
    MropeMxAppendRegionDevice(headDim * static_cast<uint32_t>(MROPE_MX_FP32_BYTES), cursor,
                              candidate.qGammaOffsetBytes);
    MropeMxAppendRegionDevice(headDim * static_cast<uint32_t>(MROPE_MX_FP32_BYTES), cursor,
                              candidate.kGammaOffsetBytes);
    MropeMxAppendRegionDevice(kvHeadNum * headDim * static_cast<uint32_t>(MROPE_MX_FP32_BYTES), cursor,
                              candidate.vScaleOffsetBytes);
    MropeMxAppendRegionDevice((headDim / 2U) * static_cast<uint32_t>(MROPE_MX_INT32_BYTES), cursor,
                              candidate.gatherIndexOffsetBytes);
    MropeMxAppendRegionDevice(
        static_cast<uint32_t>(MROPE_MX_POSITION_WINDOW_TOKENS) * 3U * static_cast<uint32_t>(MROPE_MX_INT32_BYTES),
        cursor, candidate.positionOffsetBytes);

    for (uint32_t slot = 0U; slot < 2U; ++slot) {
        const uint32_t slotBegin = MropeMxAlignDevice(cursor);
        candidate.inputSlotOffsetBytes[slot] = slotBegin;
        MropeMxAppendRegionDevice(tokenTile * headCount * headDim * static_cast<uint32_t>(MROPE_MX_BF16_BYTES), cursor,
                                  candidate.inputQkvOffsetBytes[slot]);
        MropeMxAppendRegionDevice(tokenTile * 3U * headDim * static_cast<uint32_t>(MROPE_MX_FP32_BYTES), cursor,
                                  candidate.inputRawCosSinOffsetBytes[slot]);
        MropeMxAppendRegionDevice(tokenTile * static_cast<uint32_t>(MROPE_MX_INT32_BYTES), cursor,
                                  candidate.inputSlotMappingOffsetBytes[slot]);
        cursor = MropeMxAlignDevice(cursor);
        if (slot == 0U) {
            candidate.inputSlotBytes = cursor - slotBegin;
        }
    }

    MropeMxAppendRegionDevice(tokenTile * kvHeadNum * headDim * static_cast<uint32_t>(MROPE_MX_FP8_BYTES), cursor,
                              candidate.vDataOffsetBytes, &candidate.vDataBytes);
    MropeMxAppendRegionDevice(tokenTile * kvHeadNum * headDim * static_cast<uint32_t>(MROPE_MX_FP8_BYTES), cursor,
                              candidate.kDataOffsetBytes);
    const uint32_t kScaleTokenBytes = MropeMxAlignDevice(kvHeadNum * static_cast<uint32_t>(MROPE_MX_SCALE_COUNT_D128) *
                                                         static_cast<uint32_t>(MROPE_MX_FP8_BYTES));
    MropeMxAppendRegionDevice(tokenTile * kScaleTokenBytes, cursor, candidate.kScaleOffsetBytes);
    cursor = MropeMxAlignDevice(cursor);
    candidate.kSlotBytes = cursor - candidate.kDataOffsetBytes;

    MropeMxAppendRegionDevice(tokenTile * qHeadNum * headDim * static_cast<uint32_t>(MROPE_MX_FP8_BYTES), cursor,
                              candidate.qDataOffsetBytes);
    MropeMxAppendRegionDevice(tokenTile * qHeadNum * static_cast<uint32_t>(MROPE_MX_SCALE_COUNT_D128), cursor,
                              candidate.qScaleOffsetBytes);
    cursor = MropeMxAlignDevice(cursor);
    candidate.qSlotBytes = cursor - candidate.qDataOffsetBytes;
    const uint32_t fullBatchCount = qHeadNum / static_cast<uint32_t>(MROPE_MX_ROW_BATCH_ROWS);
    const uint32_t scratchBatchCount = fullBatchCount == 0U ? 1U : fullBatchCount;
    MropeMxAppendRegionDevice(scratchBatchCount * static_cast<uint32_t>(MROPE_MX_ROW_BATCH_SCRATCH_BYTES), cursor,
                              candidate.mxScratchOffsetBytes, &candidate.mxScratchBytes);
    candidate.totalBytes = cursor;
    layout = candidate;
}

__aicore__ inline void MakeMropeMxQGlobalTileWaveUbLayoutDevice(uint32_t tokenTile, uint32_t qHeadNum,
                                                                uint32_t kvHeadNum, uint32_t headDim,
                                                                MropeMxUbLayout &layout)
{
    MakeMropeMxUbLayoutDevice(tokenTile, qHeadNum, kvHeadNum, headDim, layout);
    const uint32_t fullBatchCount = qHeadNum / static_cast<uint32_t>(MROPE_MX_ROW_BATCH_ROWS);
    const uint32_t currentScratchBatchCount = fullBatchCount == 0U ? 1U : fullBatchCount;
    const bool hasQTail = (qHeadNum % static_cast<uint32_t>(MROPE_MX_ROW_BATCH_ROWS)) ==
                          static_cast<uint32_t>(MROPE_MX_ROW_BATCH_ROWS / 2U);
    uint32_t requiredScratchBatchCount = fullBatchCount * tokenTile;
    if (hasQTail) {
        const uint32_t tailWaveBatchCount = (tokenTile + 1U) / 2U;
        requiredScratchBatchCount += tailWaveBatchCount;
    }
    if (requiredScratchBatchCount < currentScratchBatchCount) {
        requiredScratchBatchCount = currentScratchBatchCount;
    }
    const uint32_t additionalScratchBytes = (requiredScratchBatchCount - currentScratchBatchCount) *
                                            static_cast<uint32_t>(MROPE_MX_ROW_BATCH_SCRATCH_BYTES);
    layout.mxScratchBytes += additionalScratchBytes;
    layout.totalBytes += additionalScratchBytes;
}
#endif

#undef QKV_MROPE_MX_HOST_DEVICE

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_LAYOUT_H_
