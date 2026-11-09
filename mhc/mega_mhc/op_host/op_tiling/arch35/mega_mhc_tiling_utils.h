/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_UTILS_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_UTILS_H
#include <cstdint>
#include "../../../op_kernel/arch35/mega_mhc_tiling_key.h"

namespace optiling::mega_mhc {
constexpr uint32_t kHeadCount = 4U;
constexpr uint32_t kProjectionWidth = 24U;
constexpr uint32_t kMemoryFp32Bytes = sizeof(float);
// Both supported input types, FP16 and BF16, occupy 16 bits.
constexpr uint32_t kInputElementBytes = sizeof(uint16_t);
constexpr uint32_t kUbAlignmentBytes = 256U;
constexpr uint32_t kFp32VectorElements = 64U;
constexpr uint32_t kMaximumReductionRows = 48U;
constexpr uint32_t kReductionRowAlignment = 8U;
constexpr uint32_t kDmaScratchBytes = 8U;
constexpr uint32_t kCoefficientTileElements = 256U;
constexpr uint32_t kDoubleBufferSlots = 2U;
constexpr uint32_t kTripleBufferSlots = 3U;
constexpr uint32_t kEpilogueScalarBufferCount = 3U;
constexpr uint32_t kBsHfInputSlots = 2U;
constexpr uint32_t kBsHfOutputSlots = 2U;
constexpr uint32_t kBsHfScalarElements = 4U;
constexpr uint32_t kBsHfGateElements = 32U;
constexpr uint32_t kMaximumBsHfBatch = 4U;
constexpr uint32_t kMinimumBsHfBatch = 2U;
constexpr uint32_t kBsHfBatchDivisor = 2U;
constexpr uint32_t kRowTileInputBytesPerElement = 4U;
constexpr uint32_t kRowTileHeadsBytesPerElement = 16U;
constexpr uint32_t kRowTileWeightBytesPerElement = 8U;
constexpr uint32_t kRowTileFixedAllocations = 5U;
constexpr uint32_t kRowTileRmsShardElements = 8U;
constexpr uint32_t kRowTileExtraAllocations = 2U;
constexpr uint32_t kRowTileBytesPerElement = 28U;
constexpr uint32_t kRowTileAlignmentElements = 128U;
constexpr uint32_t kRowTileAlignmentOffset = 127U;

/** Bound the separate Pre/RMS1 epilogue batch to leave room for complete rows and coefficients. */
inline uint32_t CalcReductionRows(uint32_t rowLimit)
{
    return rowLimit < kMaximumReductionRows ? rowLimit : kMaximumReductionRows;
}

/** Round each live UB allocation to 256B: DMA alignment plus one full FP32 register for small arrays. */
inline uint64_t CalcUbAllocSize(uint64_t bytes)
{
    return (bytes + kUbAlignmentBytes - 1U) / kUbAlignmentBytes * kUbAlignmentBytes;
}

/** External pre coefficients occupy their own aligned UB allocation only when supplied. */
inline uint64_t CalcShiftedPreMixUbSize(uint32_t rows, bool hasShiftedPrevMix)
{
    const uint64_t alignedRows =
        (static_cast<uint64_t>(rows) + kReductionRowAlignment - 1U) / kReductionRowAlignment * kReductionRowAlignment;
    return hasShiftedPrevMix ? CalcUbAllocSize(alignedRows * kHeadCount * kMemoryFp32Bytes) : 0U;
}

/** Host-owned BS HF32 epilogue layout; inputBytes includes both input slots. */
struct BsHfEpilogueTiling {
    uint32_t batch = 0U;
    uint32_t inputBytes = 0U;
    uint64_t ubBytes = 0U;
};

/** Count all aligned live buffers: 2*batch*K inputs, two D outputs, gamma, partials, gates and scalar scratch. */
inline uint64_t CalcBsHfEpilogueUbSize(uint32_t d, uint32_t rows, uint32_t alignedRows, uint32_t partialStride,
                                       uint32_t kGrid, uint32_t batch, bool hasShiftedPrevMix = false)
{
    const uint64_t k = static_cast<uint64_t>(kHeadCount) * d;
    return CalcUbAllocSize(static_cast<uint64_t>(kBsHfInputSlots) * batch * k * kInputElementBytes) +
           kBsHfOutputSlots * CalcUbAllocSize(static_cast<uint64_t>(d) * kInputElementBytes) +
           CalcUbAllocSize(static_cast<uint64_t>(d) * kMemoryFp32Bytes) +
           CalcUbAllocSize((static_cast<uint64_t>(rows) * partialStride + kFp32VectorElements) * kMemoryFp32Bytes) +
           CalcUbAllocSize(static_cast<uint64_t>(alignedRows) * kGrid * kMemoryFp32Bytes) +
           CalcUbAllocSize(static_cast<uint64_t>(alignedRows) * kProjectionWidth * kMemoryFp32Bytes) +
           CalcUbAllocSize(kBsHfScalarElements * kMemoryFp32Bytes) +
           CalcUbAllocSize(kBsHfGateElements * kMemoryFp32Bytes) +
           CalcUbAllocSize(kBsHfScalarElements * kMemoryFp32Bytes) +
           CalcShiftedPreMixUbSize(alignedRows, hasShiftedPrevMix);
}

/** Select the largest validated batch b in {4,2} with aligned live UB(b)<=availableUb; zero keeps the row path. */
inline BsHfEpilogueTiling CalcBsHfEpilogueTiling(uint64_t totalRows, uint32_t d, uint32_t rows, uint32_t alignedRows,
                                                 uint32_t partialStride, uint32_t kGrid, uint64_t availableUb,
                                                 bool hasShiftedPrevMix = false)
{
    constexpr uint32_t kMinimumBatchedRows = 3072U;
    BsHfEpilogueTiling result;
    if (totalRows < kMinimumBatchedRows || d == 0U || rows == 0U || kGrid != 1U) {
        return result;
    }
    for (uint32_t batch = kMaximumBsHfBatch; batch >= kMinimumBsHfBatch; batch /= kBsHfBatchDivisor) {
        const uint64_t bytes =
            CalcBsHfEpilogueUbSize(d, rows, alignedRows, partialStride, kGrid, batch, hasShiftedPrevMix);
        if (bytes <= availableUb) {
            result.batch = batch;
            result.inputBytes = kBsHfInputSlots * batch * kHeadCount * d * kInputElementBytes;
            result.ubBytes = bytes;
            return result;
        }
    }
    return result;
}

/** Count the live full-row producer buffers, including double input and coefficient slots.
 * NoMid non-wave aliases Post with three residual slots; Mid/Wave use their planned residual slots.
 * Wave RMS0 writes back into its input slot, so it needs no separate x1 allocation.
 * RMS0 bypass retains this conservative layout; its gamma0 allocation is reserved but never loaded.
 */
inline uint64_t CalcProducerUbSize(uint32_t d, bool hasMid, bool wavePipeline = false, uint32_t gammaElements = 0U,
                                   uint32_t residualSlots = kDoubleBufferSlots)
{
    const uint64_t k = static_cast<uint64_t>(kHeadCount) * d;
    const uint32_t liveResidualSlots = !hasMid && !wavePipeline ? kTripleBufferSlots : residualSlots;
    return CalcUbAllocSize(static_cast<uint64_t>(kDoubleBufferSlots) * kInputElementBytes * d) +
           CalcUbAllocSize(static_cast<uint64_t>(kInputElementBytes) * liveResidualSlots * k) +
           CalcUbAllocSize(static_cast<uint64_t>(kMemoryFp32Bytes) * d) +
           CalcUbAllocSize(kCoefficientTileElements * kMemoryFp32Bytes) + CalcUbAllocSize(kDmaScratchBytes) +
           (hasMid ?
                CalcUbAllocSize(static_cast<uint64_t>(kMemoryFp32Bytes) * (gammaElements == 0U ? k : gammaElements)) :
                0U);
}

/** Exchange one residual prefetch slot for a full, once-loaded Mid gamma only when it changes the fit. */
inline bool CanUseSingleResidualForMid(uint32_t d, uint64_t availableUb)
{
    const uint32_t k = kHeadCount * d;
    return CalcProducerUbSize(d, true, false, k) > availableUb &&
           CalcProducerUbSize(d, true, false, k, 1U) <= availableUb;
}

/** BS Mid producer: double inputs, one/two aliased residual slots and in-place RMS0 allow complete gamma2 caching. */
inline uint64_t CalcCachedGammaProducerUbSize(uint32_t d, uint32_t residualSlots = 1U)
{
    const uint64_t k = static_cast<uint64_t>(kHeadCount) * d;
    return CalcUbAllocSize(static_cast<uint64_t>(kDoubleBufferSlots) * kInputElementBytes * d) +
           CalcUbAllocSize(static_cast<uint64_t>(kInputElementBytes) * k * residualSlots) +
           CalcUbAllocSize(static_cast<uint64_t>(kMemoryFp32Bytes) * d) +
           CalcUbAllocSize(static_cast<uint64_t>(kMemoryFp32Bytes) * k) +
           CalcUbAllocSize(kCoefficientTileElements * kMemoryFp32Bytes) + CalcUbAllocSize(kDmaScratchBytes);
}

/** Materialized-row projection owns row statistics plus packed input and FP32 transfer half-tiles. */
inline uint64_t CalcProjectionUbSize(uint32_t l1K, uint32_t projectionRows, uint32_t stageSlots = 1U,
                                     uint32_t stageRowStride = 0U)
{
    const uint64_t stageElements = static_cast<uint64_t>(projectionRows) * l1K;
    const uint64_t transferElements =
        static_cast<uint64_t>(stageRowStride == 0U ? projectionRows : stageRowStride) * l1K;
    return CalcUbAllocSize(projectionRows * kMemoryFp32Bytes) +
           CalcUbAllocSize(transferElements * kMemoryFp32Bytes * stageSlots) +
           CalcUbAllocSize(stageElements * kInputElementBytes) + CalcUbAllocSize(stageElements * kInputElementBytes);
}

struct ProjectionUbTilingInfo {
    uint32_t rowStride = 0U;
    uint32_t stageSlots = 1U;
    uint64_t bytes = 0U;
};

/** Select padded projection rows and FP32 stage slots with the concurrent producer live in UB. */
inline ProjectionUbTilingInfo CalcProjectionUbTiling(uint32_t l1K, uint32_t rows, uint64_t availableUb,
                                                     uint64_t producerBytes, bool packedNz)
{
    const bool padRows = packedNz && producerBytes + CalcProjectionUbSize(l1K, rows, 1U, rows + 1U) <= availableUb;
    const uint32_t stride = rows + static_cast<uint32_t>(padRows);
    const uint32_t slots = producerBytes + CalcProjectionUbSize(l1K, rows, kDoubleBufferSlots, stride) <= availableUb ?
                               kDoubleBufferSlots :
                               1U;
    return {stride, slots, CalcProjectionUbSize(l1K, rows, slots, stride)};
}

/** Choose the fewest gamma loads whose aligned, equal-sized chunks fit: G=align64(ceil(4D/count)). */
inline uint32_t CalcMidGammaTile(uint32_t d, uint64_t availableUb)
{
    const uint32_t k = kHeadCount * d;
    for (uint32_t count = 1U; count <= k / kFp32VectorElements; ++count) {
        const uint32_t tile =
            ((k + count - 1U) / count + kFp32VectorElements - 1U) / kFp32VectorElements * kFp32VectorElements;
        if (CalcProducerUbSize(d, true, false, tile) <= availableUb) {
            return tile;
        }
    }
    return 0U;
}

/** Epilogue owns one x2/hIn row plus a host-sized batch of projection partials. */
inline uint64_t CalcEpilogueUbSize(uint32_t d, uint32_t partialGroups, uint32_t partialStride, uint32_t reduceRows = 1U,
                                   uint32_t kGrid = 1U, bool reserveInput = false, bool hasShiftedPrevMix = false)
{
    uint64_t partialBytes =
        (static_cast<uint64_t>(partialGroups) * reduceRows * partialStride + kFp32VectorElements) * kMemoryFp32Bytes;
    const uint64_t inputRowBytes = static_cast<uint64_t>(kHeadCount) * d * kInputElementBytes;
    if (reserveInput && partialBytes < inputRowBytes) {
        partialBytes = inputRowBytes;
    }
    return CalcUbAllocSize(inputRowBytes) + kDoubleBufferSlots * CalcUbAllocSize(d * kInputElementBytes) +
           CalcUbAllocSize(d * kMemoryFp32Bytes) + kEpilogueScalarBufferCount * kUbAlignmentBytes +
           CalcUbAllocSize(static_cast<uint64_t>((reduceRows + kReductionRowAlignment - 1U) / kReductionRowAlignment *
                                                 kReductionRowAlignment) *
                           kProjectionWidth * kMemoryFp32Bytes) +
           CalcUbAllocSize(static_cast<uint64_t>((reduceRows + kReductionRowAlignment - 1U) / kReductionRowAlignment *
                                                 kReductionRowAlignment) *
                           kGrid * kMemoryFp32Bytes) +
           CalcUbAllocSize(partialBytes) + CalcShiftedPreMixUbSize(reduceRows, hasShiftedPrevMix);
}
/** Select the largest eight-row batch fitting the exact epilogue live allocation; small M keeps its tail. */
inline uint32_t FitReductionRows(uint32_t rowLimit, uint32_t d, uint32_t groups, uint32_t stride, uint32_t kGrid,
                                 uint64_t availableUb, bool hasShiftedPrevMix = false)
{
    uint32_t rows = CalcReductionRows(rowLimit);
    while (rows > 0U && CalcEpilogueUbSize(d, groups, stride, rows, kGrid, false, hasShiftedPrevMix) > availableUb) {
        rows =
            rows > kReductionRowAlignment ? (rows - 1U) / kReductionRowAlignment * kReductionRowAlignment : rows - 1U;
    }
    return rows;
}

/** Balance R per-core rows over B=ceil(R/capacity) batches without increasing the capacity bound.
 * Batch rows=min(capacity,align8(ceil(R/B))); a zero capacity or workload has no valid batch.
 */
inline uint32_t BalanceReductionRows(uint32_t rowsPerCore, uint32_t capacityRows)
{
    if (capacityRows == 0U || rowsPerCore == 0U) {
        return 0U;
    }
    const uint64_t batches = (static_cast<uint64_t>(rowsPerCore) + capacityRows - 1U) / capacityRows;
    const uint32_t balancedRows =
        static_cast<uint32_t>(((rowsPerCore + batches - 1U) / batches + kReductionRowAlignment - 1U) /
                              kReductionRowAlignment * kReductionRowAlignment);
    return balancedRows < capacityRows ? balancedRows : capacityRows;
}

/** Two ping-pong D tiles own input T[2,D], heads T[2,4,D] and weights FP32[2,D].
 * Coefficients and statistics are separate register-padded allocations: peak=28*tile+1280 bytes.
 */
inline uint64_t CalcRowTileProducerUbSize(uint32_t tile)
{
    return CalcUbAllocSize(static_cast<uint64_t>(kRowTileInputBytesPerElement) * tile) +
           CalcUbAllocSize(static_cast<uint64_t>(kRowTileHeadsBytesPerElement) * tile) +
           CalcUbAllocSize(static_cast<uint64_t>(kRowTileWeightBytesPerElement) * tile) +
           kRowTileFixedAllocations * kUbAlignmentBytes;
}

/** Add the one-row ordered projection partials, RMS shards and immutable gate parameters to the tiled epilogue. */
inline uint64_t CalcRowTileEpilogueUbSize(uint32_t tile, uint32_t groups, uint32_t stride, uint32_t kGrid,
                                          bool hasShiftedPrevMix = false)
{
    return CalcRowTileProducerUbSize(tile) +
           CalcUbAllocSize((static_cast<uint64_t>(groups) * stride + kFp32VectorElements) * kMemoryFp32Bytes) +
           CalcUbAllocSize(static_cast<uint64_t>(kRowTileRmsShardElements) * kGrid * kMemoryFp32Bytes) +
           kRowTileExtraAllocations * kUbAlignmentBytes + CalcShiftedPreMixUbSize(1U, hasShiftedPrevMix);
}

struct RowTilingInfo {
    uint32_t enabled = 0U;
    uint32_t tile = 0U;
    uint32_t count = 0U;
    uint32_t tail = 0U;
};

/** Retain the complete-row path when it fits. Otherwise balance D over the fewest capacity-bounded 128-aligned tiles.
 * Capacity=floor((UB-fixed)/28/128)*128; count=ceil(D/capacity); tile=align128(ceil(D/count)).
 * All loop counts and valid tail lengths are Host results; no shape-specific constants or kernel tiling search.
 */
inline RowTilingInfo CalcRowTiling(uint32_t d, bool hasMid, uint32_t groups, uint32_t stride, uint32_t kGrid,
                                   uint64_t availableUb, bool hasShiftedPrevMix = false)
{
    if (d == 0U) {
        return {};
    }
    if (CalcProducerUbSize(d, hasMid, false, hasMid ? kFp32VectorElements : 0U) <= availableUb &&
        CalcEpilogueUbSize(d, groups, stride, 1U, kGrid, false, hasShiftedPrevMix) <= availableUb) {
        return {0U, d, 1U, d};
    }
    const uint64_t fixed = CalcRowTileEpilogueUbSize(0U, groups, stride, kGrid, hasShiftedPrevMix);
    if (availableUb < fixed + kRowTileBytesPerElement * kRowTileAlignmentElements) {
        return {};
    }
    const uint32_t capacity = static_cast<uint32_t>(
        (availableUb - fixed) / (kRowTileBytesPerElement * kRowTileAlignmentElements) * kRowTileAlignmentElements);
    const uint32_t count = (d + capacity - 1U) / capacity;
    const uint32_t tile =
        ((d + count - 1U) / count + kRowTileAlignmentOffset) / kRowTileAlignmentElements * kRowTileAlignmentElements;
    return {1U, tile, count, d - (count - 1U) * tile};
}

} // namespace optiling::mega_mhc
#endif
