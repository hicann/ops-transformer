/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../op_host/op_tiling/arch35/mega_mhc_cube_tiling.h"
#include "../../../op_host/op_tiling/arch35/mega_mhc_tiling.h"
#include "../../../op_host/op_tiling/arch35/mega_mhc_tiling_utils.h"

#include <algorithm>
#include <initializer_list>
#include <gtest/gtest.h>
#include <cstring>
#include "securec.h"
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace optiling::mega_mhc;
#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            return __LINE__; \
        } \
    } while (false)

/** Exercise the actual host tiling calculation: exact M/K coverage, partial ownership, tails and physical capacities.
 */
static int CheckCubeTilingInvariants()
{
    constexpr uint64_t kUbBudget = 248U * 1024U;
    // Wave RMS0 aliases its input. Its exact live layout must permit padded direct transfer at large D.
    REQUIRE(CalcProducerUbSize(7168U, false, true) == 173312U);
    const auto largeWaveUb = CalcProjectionUbTiling(128U, 64U, kUbBudget, CalcProducerUbSize(7168U, false, true), true);
    REQUIRE(largeWaveUb.rowStride == 65U && largeWaveUb.stageSlots == 1U);
    REQUIRE(largeWaveUb.bytes + CalcProducerUbSize(7168U, false, true) <= kUbBudget);
    const auto doubleWaveUb =
        CalcProjectionUbTiling(128U, 64U, kUbBudget, CalcProducerUbSize(6144U, false, true), true);
    REQUIRE(doubleWaveUb.rowStride == 65U && doubleWaveUb.stageSlots == 2U);
    REQUIRE(doubleWaveUb.bytes + CalcProducerUbSize(6144U, false, true) <= kUbBudget);
    REQUIRE(!CanUseSingleResidualForMid(5120U, kUbBudget));
    REQUIRE(!CanUseSingleResidualForMid(6144U, kUbBudget));
    REQUIRE(CanUseSingleResidualForMid(7168U, kUbBudget));
    REQUIRE(!CanUseSingleResidualForMid(7168U, 128U * 1024U));
    REQUIRE(CalcMidGammaTile(6144U, kUbBudget) == 24576U);
    REQUIRE(CalcMidGammaTile(7168U, kUbBudget) == 14336U);
    REQUIRE(CalcProducerUbSize(6144U, true, false, 4U * 6144U, 1U) == 197888U);
    REQUIRE(CalcProducerUbSize(7168U, true, false, 4U * 7168U, 1U) == 230656U);
    for (uint32_t d : {5120U, 6144U, 7168U}) {
        const bool singleResidual = CanUseSingleResidualForMid(d, kUbBudget);
        const uint32_t gammaTile = singleResidual ? 4U * d : CalcMidGammaTile(d, kUbBudget);
        REQUIRE(CalcProducerUbSize(d, true, false, gammaTile, singleResidual ? 1U : 2U) <= kUbBudget);
    }
    // Wave costs may favor fewer handoffs only when the actual concurrent UB layout permits an epilogue.
    for (uint32_t d : {2560U, 5120U}) {
        CubeTilingParams request;
        request.rows = d == 2560U ? 4096U : 1024U;
        request.d = d;
        request.k = 4U * d;
        request.q = 24U;
        request.ubBytes = kUbBudget;
        request.packedNz = true;
        request.aicCount = 32U;
        request.aivCount = 64U;
        request.l1ABytes = request.l1BBytes = 256U * 1024U;
        request.l0ABytes = request.l0BBytes = 64U * 1024U;
        request.l0CBytes = 128U * 1024U;
        request.wavePipeline = true;
        CubeTilingInfo selected;
        REQUIRE(CalcCubeTiling(request, selected));
        auto referenceRequest = request;
        referenceRequest.ubBytes = 0U;
        CubeTilingInfo reference;
        REQUIRE(CalcCubeTiling(referenceRequest, reference));
        const uint64_t producerBytes = CalcProducerUbSize(d, false, true);
        const auto ub = CalcProjectionUbTiling(selected.l1K, selected.mTile / 2U, kUbBudget, producerBytes, true);
        const uint32_t epilogueRows =
            producerBytes + ub.bytes < kUbBudget ?
                FitReductionRows(selected.mTile / (2U * selected.kGrid), d, selected.partialGroupCount,
                                 selected.fixpipeRowStride, selected.kGrid, kUbBudget - producerBytes - ub.bytes) :
                0U;
        if (epilogueRows != 0U) {
            REQUIRE(selected.mWaveCount < reference.mWaveCount);
            REQUIRE(producerBytes + ub.bytes +
                        CalcEpilogueUbSize(d, selected.partialGroupCount, selected.fixpipeRowStride, epilogueRows,
                                           selected.kGrid) <=
                    kUbBudget);
        } else {
            REQUIRE(selected.mTile == reference.mTile && selected.mGrid == reference.mGrid);
            REQUIRE(selected.kGrid == reference.kGrid && selected.partialGroupK == reference.partialGroupK);
        }
        // The decision follows actual UB capacity, including the same D on a different resource budget.
        request.ubBytes = epilogueRows != 0U ? 128U * 1024U : 384U * 1024U;
        CubeTilingInfo resized;
        REQUIRE(CalcCubeTiling(request, resized));
        if (epilogueRows != 0U) {
            REQUIRE(resized.mTile == reference.mTile && resized.mGrid == reference.mGrid);
            REQUIRE(resized.kGrid == reference.kGrid);
        } else {
            const auto largerUb =
                CalcProjectionUbTiling(resized.l1K, resized.mTile / 2U, request.ubBytes, producerBytes, true);
            REQUIRE(producerBytes + largerUb.bytes < request.ubBytes);
            REQUIRE(FitReductionRows(resized.mTile / (2U * resized.kGrid), d, resized.partialGroupCount,
                                     resized.fixpipeRowStride, resized.kGrid,
                                     request.ubBytes - producerBytes - largerUb.bytes) > 0U);
        }
    }
    // Partial-unit balancing must retain short-but-nonempty K lanes, independent of the accumulation bound.
    for (uint64_t d : {5120U, 7168U}) {
        for (auto mode : {ProjectionMode::FP32, ProjectionMode::HF32}) {
            CubeTilingParams r;
            r.schedule = ScheduleMode::MK;
            r.mode = mode;
            r.rows = 128U;
            r.k = 4U * d;
            r.d = static_cast<uint32_t>(d);
            r.ubBytes = kUbBudget;
            r.packedNz = true;
            r.q = 24U;
            r.aicCount = 32U;
            r.aivCount = 64U;
            r.l1ABytes = r.l1BBytes = 256U * 1024U;
            r.l0ABytes = r.l0BBytes = 64U * 1024U;
            r.l0CBytes = 128U * 1024U;
            CubeTilingInfo p;
            REQUIRE(CalcCubeTiling(r, p));
            REQUIRE(p.activeAic == r.aicCount);
            REQUIRE(p.lastKCoreElements > 0U);
            // Reuse planning must retain complete core ownership and nonempty K shards.
            r.rows = 1024U;
            r.maxReuseMStages = 8U;
            REQUIRE(CalcCubeTiling(r, p));
            REQUIRE(p.activeAic == r.aicCount);
            REQUIRE(p.lastKCoreElements > 0U);
        }
    }
    for (auto schedule : {ScheduleMode::MK, ScheduleMode::BS}) {
        for (uint64_t m : {1U, 17U, 33U, 255U, 256U, 257U, 512U, 1025U, 3072U, 8192U, 8193U}) {
            for (uint64_t d : {16U, 48U, 128U, 1008U, 2560U, 2576U, 2816U, 5120U, 6144U, 7168U, 7424U, 8192U}) {
                for (auto mode : {ProjectionMode::FP32, ProjectionMode::HF32}) {
                    CubeTilingParams r;
                    r.schedule = schedule;
                    r.rows = m;
                    r.k = 4U * d;
                    r.d = static_cast<uint32_t>(d);
                    r.ubBytes = kUbBudget;
                    r.packedNz = true;
                    r.q = 24U;
                    r.aicCount = 32U;
                    r.aivCount = 64U;
                    r.l1ABytes = r.l1BBytes = 256U * 1024U;
                    r.l0ABytes = r.l0BBytes = 64U * 1024U;
                    r.l0CBytes = 128U * 1024U;
                    r.mode = mode;
                    CubeTilingInfo p;
                    REQUIRE(CalcCubeTiling(r, p));
                    if (schedule == ScheduleMode::BS) {
                        REQUIRE(p.kGrid == 1U);
                    }
                    REQUIRE(p.activeAic == p.mGrid * p.kGrid && p.activeAic <= r.aicCount);
                    REQUIRE((p.mWaveCount - 1ULL) * p.rowsPerWave + p.lastWaveRows == m);
                    REQUIRE((p.lastWaveMGrid - 1ULL) * p.mTile + p.lastMCoreRows == p.lastWaveRows);
                    const uint64_t largeBeforeLast = std::min(p.kGrid - 1U, p.largeKCoreCount);
                    REQUIRE(largeBeforeLast * p.splitK + (p.kGrid - 1U - largeBeforeLast) * p.shortSplitK +
                                p.lastKCoreElements ==
                            r.k);
                    REQUIRE(p.lastKCoreElements > 0U && p.lastKCoreElements <= p.splitK);
                    REQUIRE(p.partialGroupK < 2048U);
                    REQUIRE(2ULL * p.l0ASlotElements * 4U <= r.l0ABytes);
                    REQUIRE(2ULL * p.l0BSlotElements * 4U <= r.l0BBytes);
                    REQUIRE(2ULL * p.l0CSlotElements * 4U <= r.l0CBytes);
                    REQUIRE(CalcProjectionUbSize(p.l1K, (p.mTile + 1U) / 2U) <= 248U * 1024U);
                }
            }
        }
    }
    // One MK tiling calculation covers both Cube precisions and all M-grid counts.
    for (uint64_t m : {16U, 32U, 64U, 128U, 256U, 512U, 1024U, 2048U, 3072U, 4096U, 6144U, 8192U}) {
        for (uint64_t d : {2560U, 5120U, 6144U, 7168U}) {
            CubeTilingParams request;
            request.schedule = ScheduleMode::MK;
            request.rows = m;
            request.k = 4U * d;
            request.d = static_cast<uint32_t>(d);
            request.ubBytes = kUbBudget;
            request.packedNz = true;
            request.q = 24U;
            request.aicCount = 32U;
            request.aivCount = 64U;
            request.l1ABytes = request.l1BBytes = 256U * 1024U;
            request.l0ABytes = request.l0BBytes = 64U * 1024U;
            request.l0CBytes = 128U * 1024U;
            CubeTilingInfo fp32Tiling;
            CubeTilingInfo hf32Tiling;
            request.mode = ProjectionMode::FP32;
            REQUIRE(CalcCubeTiling(request, fp32Tiling));
            request.mode = ProjectionMode::HF32;
            REQUIRE(CalcCubeTiling(request, hf32Tiling));
            REQUIRE(fp32Tiling.mGrid == hf32Tiling.mGrid);
            REQUIRE(fp32Tiling.kGrid == hf32Tiling.kGrid);
            REQUIRE(fp32Tiling.mTile == hf32Tiling.mTile);
            REQUIRE(fp32Tiling.splitK == hf32Tiling.splitK);
            REQUIRE(fp32Tiling.partialGroupK == hf32Tiling.partialGroupK);
            REQUIRE(fp32Tiling.l1K == hf32Tiling.l1K);
            REQUIRE(fp32Tiling.activeAic == hf32Tiling.activeAic);
            REQUIRE(fp32Tiling.activeAiv == hf32Tiling.activeAiv);
            if (m >= 64U && d == 2560U) {
                REQUIRE(hf32Tiling.mGrid > 1U);
                REQUIRE(hf32Tiling.activeAic == 32U && hf32Tiling.activeAiv == 64U);
            }
        }
    }
    const bool midModes[] = {false, true};
    const uint64_t rows[] = {1,   2,   15,   16,   17,   31,   32,   33,   63,   64,   160,  192,          257,
                             512, 768, 1024, 1536, 1537, 2048, 3071, 3072, 3073, 4096, 6145, 4294967295ULL};
    const uint64_t widths[] = {2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864,
                               5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168};
    const uint64_t coreCounts[] = {1, 7, 24, 32, 48};
    for (auto m : rows) {
        for (auto d : widths) {
            for (auto cores : coreCounts) {
                for (uint32_t variant = 0; variant < 4; ++variant) {
                    const uint32_t mode = variant & 1U;
                    CubeTilingParams request;
                    request.rows = m;
                    const bool hasMid = (variant & 2U) != 0U;
                    request.k = 4 * d;
                    request.d = static_cast<uint32_t>(d);
                    request.ubBytes = kUbBudget;
                    request.packedNz = true;
                    request.q = 24;
                    request.aicCount = cores;
                    request.aivCount = 2 * cores;
                    request.l1ABytes = 256 * 1024;
                    request.l1BBytes = 256 * 1024;
                    request.l0ABytes = 64 * 1024;
                    request.l0BBytes = 64 * 1024;
                    request.l0CBytes = 128 * 1024;
                    request.mode = static_cast<ProjectionMode>(mode);
                    request.wavePipeline =
                        !hasMid &&
                        ((m == 3072 && cores >= 4 && cores % 2 == 0) ||
                         (cores == 32 && (m == 512 || m == 768 || m == 1024 || m == 1536 || m == 2048 || m == 4096)));
                    CubeTilingInfo p;
                    if (!CalcCubeTiling(request, p)) {
                        REQUIRE(request.wavePipeline);
                        request.wavePipeline = false;
                        REQUIRE(CalcCubeTiling(request, p));
                    }
                    // K-tail compaction can invalidate complete-row ownership; the host then uses its non-wave tiling.
                    if (request.wavePipeline &&
                        (p.mTile % (2U * p.kGrid) != 0U || p.mWaveCount <= 1U || p.lastKCoreL1FullLoops < 2U)) {
                        request.wavePipeline = false;
                        REQUIRE(CalcCubeTiling(request, p));
                    }
                    // The unified grid search must be independent of Cube precision.
                    if (!request.wavePipeline && request.mode == ProjectionMode::FP32) {
                        CubeTilingParams referenceRequest = request;
                        referenceRequest.mode = ProjectionMode::HF32;
                        CubeTilingInfo referenceTiling;
                        REQUIRE(CalcCubeTiling(referenceRequest, referenceTiling));
                        REQUIRE(p.mGrid == referenceTiling.mGrid);
                        REQUIRE(p.kGrid == referenceTiling.kGrid && p.partialGroupK == referenceTiling.partialGroupK);
                        REQUIRE(p.l1K == referenceTiling.l1K);
                        REQUIRE(p.activeAic >= referenceTiling.activeAic);
                    }
                    REQUIRE(p.activeAic <= cores && p.activeAic == p.mGrid * p.kGrid);
                    REQUIRE(p.activeAiv == 2 * p.activeAic);
                    if (request.wavePipeline) {
                        auto initialRequest = request;
                        initialRequest.ubBytes = 0U;
                        CubeTilingInfo initialTiling;
                        REQUIRE(CalcCubeTiling(initialRequest, initialTiling));
                        REQUIRE(p.mWaveCount <= initialTiling.mWaveCount);
                        // Each producer batch must finish before the next wave's projection reads it.
                        const uint32_t members = p.kGrid * 2U;
                        const uint32_t producerRows = p.mTile / members;
                        REQUIRE(producerRows > 0 && producerRows * members == p.mTile);
                        const uint32_t fullProducers = p.lastWaveRows / producerRows;
                        const uint32_t tailProducerRows = p.lastWaveRows % producerRows;
                        REQUIRE(fullProducers <= p.activeAiv);
                        REQUIRE(fullProducers * producerRows + tailProducerRows == p.lastWaveRows);
                        const uint32_t ticksAvailable = p.lastKCoreL1FullLoops - 1U;
                        const uint32_t rowsPerTick = (producerRows + ticksAvailable - 1U) / ticksAvailable;
                        const uint32_t ticks = (producerRows + rowsPerTick - 1U) / rowsPerTick;
                        const uint32_t tickStride = ticksAvailable / ticks;
                        REQUIRE(tickStride > 0U);
                        REQUIRE(1U + (ticks - 1U) * tickStride < p.lastKCoreL1FullLoops);
                        const uint64_t producerUb = CalcUbAllocSize(2ULL * d * 2U) +
                                                    CalcUbAllocSize(2ULL * request.k * 2U) + CalcUbAllocSize(d * 4U) +
                                                    CalcUbAllocSize(256U * 4U) + CalcUbAllocSize(8U);
                        REQUIRE(producerUb + CalcProjectionUbSize(p.l1K, p.mTile / 2U) <= 248U * 1024U);
                    }
                    REQUIRE((uint64_t(p.mWaveCount) - 1) * p.rowsPerWave + p.lastWaveRows == m);
                    REQUIRE((p.lastWaveMGrid - 1) * p.mTile + p.lastMCoreRows == p.lastWaveRows);
                    REQUIRE(p.lastMCoreRows > 0 && p.lastMCoreRows <= p.mTile);
                    uint64_t coveredK = 0;
                    for (uint32_t shard = 0; shard < p.kGrid; ++shard) {
                        const uint32_t count = shard + 1U == p.kGrid ?
                                                   p.lastKCoreElements :
                                                   (shard < p.largeKCoreCount ? p.splitK : p.shortSplitK);
                        REQUIRE(coveredK % p.partialGroupK == 0U);
                        REQUIRE(count > 0U);
                        REQUIRE(p.splitK - count < p.partialGroupK * 2U);
                        coveredK += count;
                    }
                    REQUIRE(coveredK == request.k);
                    REQUIRE(p.shortKCoreL1FullLoops * p.l1K == p.shortSplitK || p.largeKCoreCount == p.kGrid);
                    REQUIRE(p.lastKCoreElements > 0 && p.lastKCoreElements <= p.splitK);
                    REQUIRE(p.splitK % p.partialGroupK == 0);
                    if (p.splitK >= 2048U) {
                        REQUIRE(p.partialGroupK == 1024U);
                        REQUIRE(p.partialGroupK % p.l1K == 0U);
                    } else {
                        REQUIRE(p.partialGroupK == p.splitK);
                        REQUIRE(p.partialGroupCount == p.kGrid);
                    }
                    REQUIRE(uint64_t(p.partialGroupCount) * p.partialGroupK >= request.k);
                    REQUIRE(uint64_t(p.partialGroupCount - 1) * p.partialGroupK < request.k);
                    REQUIRE(p.l1KFullLoops * p.l1K + p.l1KTail == p.splitK);
                    REQUIRE(p.lastKCoreL1FullLoops * p.l1K + p.lastKCoreL1Tail == p.lastKCoreElements);
                    REQUIRE(p.l0KLoops * p.baseK + p.l0KTail == p.l1K);
                    REQUIRE(p.baseK % 32 == 0 && p.l1K % 128 == 0);
                    REQUIRE(p.fullMmadKLoopCount * p.fullMmadKStep == p.baseK);
                    REQUIRE(p.fullMmadKLoopCount == 1U && p.fullMmadKStep == p.baseK);
                    REQUIRE(p.l0TailMmadKLoopCount == static_cast<uint32_t>(p.l0KTail != 0U));
                    REQUIRE(p.l0TailMmadKStep == p.l0KTail && p.l0TailMmadFinalKStep == p.l0KTail);
                    REQUIRE(p.tailL1TailMmadKLoopCount == static_cast<uint32_t>(p.tailL1L0Tail != 0U));
                    REQUIRE(p.tailL1TailMmadKStep == p.tailL1L0Tail && p.tailL1TailMmadFinalKStep == p.tailL1L0Tail);
                    REQUIRE(p.lastCoreTailMmadKLoopCount == static_cast<uint32_t>(p.lastCoreTailL1L0Tail != 0U));
                    REQUIRE(p.lastCoreTailMmadKStep == p.lastCoreTailL1L0Tail &&
                            p.lastCoreTailMmadFinalKStep == p.lastCoreTailL1L0Tail);
                    REQUIRE(2ULL * p.l1ASlotElements * 4 <= request.l1ABytes);
                    REQUIRE(2ULL * p.l1BSlotElements * 4 <= request.l1BBytes);
                    REQUIRE(2ULL * p.l0ASlotElements * 4 <= request.l0ABytes);
                    REQUIRE(2ULL * p.l0BSlotElements * 4 <= request.l0BBytes);
                    REQUIRE(uint64_t(p.l0CSlotElements) * 4 * 2 <= request.l0CBytes);
                    REQUIRE(p.baseM == p.nzM && p.fixpipeRowStride == p.nzN && p.nzN >= request.q);
                    for (bool mid : midModes) {
                        const uint32_t gammaTile = CalcMidGammaTile(d, 248U * 1024U);
                        REQUIRE(gammaTile > 0U && gammaTile % 64U == 0U);
                        REQUIRE(CalcProducerUbSize(d, mid, false, gammaTile) <= 248U * 1024U);
                        const uint32_t reductionRowLimit =
                            request.wavePipeline ?
                                static_cast<uint32_t>((m + p.activeAiv - 1U) / p.activeAiv) :
                                std::min(p.mTile / 2U, static_cast<uint32_t>((m + p.activeAiv - 1U) / p.activeAiv));
                        const uint32_t capacityRows = FitReductionRows(reductionRowLimit, d, p.partialGroupCount,
                                                                       p.fixpipeRowStride, p.kGrid, 248U * 1024U);
                        const uint32_t reduceRows = BalanceReductionRows(reductionRowLimit, capacityRows);
                        REQUIRE(reduceRows > 0 && reduceRows <= reductionRowLimit);
                        REQUIRE(CalcEpilogueUbSize(d, p.partialGroupCount, p.fixpipeRowStride, reduceRows, p.kGrid) <=
                                248U * 1024U);
                        REQUIRE(CalcProjectionUbSize(p.l1K, p.mTile / 2) <= 248U * 1024U);
                    }
                }
            }
        }
    }
    // Tail plans must cover each producer row exactly once, including empty producer members.
    for (uint64_t m : {513ULL, 576ULL, 640ULL, 1023ULL, 1025ULL, 2176ULL, 2560ULL, 2561ULL, 8192ULL, 8193ULL}) {
        for (auto d : widths) {
            CubeTilingParams request;
            request.rows = m;
            request.k = 4U * d;
            request.d = static_cast<uint32_t>(d);
            request.ubBytes = kUbBudget;
            request.packedNz = true;
            request.q = 24U;
            request.l1ABytes = request.l1BBytes = 256U * 1024U;
            request.l0ABytes = request.l0BBytes = 64U * 1024U;
            request.l0CBytes = 128U * 1024U;
            request.wavePipeline = true;
            CubeTilingInfo p;
            request.aicCount = 32U;
            request.aivCount = 64U;
            bool found = false;
            for (bool complete : {true, false}) {
                request.preferCompleteWaves = complete;
                if (CalcCubeTiling(request, p) && p.mTile % (2U * p.kGrid) == 0U && p.mWaveCount > 1U &&
                    p.lastKCoreL1FullLoops >= 2U &&
                    CalcProducerUbSize(d, false, true) + CalcProjectionUbSize(p.l1K, p.mTile / 2U) <= 248U * 1024U) {
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
            const uint32_t rowsPerProducer = p.mTile / (2U * p.kGrid);
            REQUIRE(p.activeAiv * rowsPerProducer == p.rowsPerWave);
            const uint32_t full = p.lastWaveRows / rowsPerProducer;
            const uint32_t tail = p.lastWaveRows % rowsPerProducer;
            uint64_t covered = (p.mWaveCount - 1ULL) * p.rowsPerWave;
            for (uint32_t member = 0U; member < p.activeAiv; ++member) {
                covered += member < full ? rowsPerProducer : (member == full ? tail : 0U);
            }
            REQUIRE(covered == m);
        }
    }
    // Reuse grids must cover all M/K tails exactly while keeping every accumulation below K=2048.
    for (uint64_t rows : {1U, 16U, 33U, 128U, 256U, 512U, 1024U, 2048U, 4096U, 8192U}) {
        for (uint64_t d : {16U, 1008U, 2560U, 2576U, 5120U, 6144U, 7168U, 8192U, 8208U, 10000U, 16368U, 16384U}) {
            for (auto mode : {ProjectionMode::FP32, ProjectionMode::HF32}) {
                CubeTilingParams r;
                r.mode = mode;
                r.rows = rows;
                r.k = 4U * d;
                r.d = static_cast<uint32_t>(d);
                r.ubBytes = kUbBudget;
                r.packedNz = true;
                r.q = 24U;
                r.aicCount = 32U;
                r.aivCount = 64U;
                r.l1ABytes = r.l1BBytes = 256U * 1024U;
                r.l0ABytes = r.l0BBytes = 64U * 1024U;
                r.l0CBytes = 256U * 1024U;
                r.maxReuseMStages = 8U;
                CubeTilingInfo p;
                REQUIRE(CalcCubeTiling(r, p));
                REQUIRE(p.partialGroupK < 2048U && p.partialGroupK % p.l1K == 0U);
                REQUIRE(p.activeAic == p.mGrid * p.kGrid && p.activeAic <= 32U);
                uint64_t coverage = 0U;
                for (uint32_t lane = 0U; lane < p.kGrid; ++lane) {
                    const uint64_t length = lane + 1U == p.kGrid ?
                                                p.lastKCoreElements :
                                                (lane < p.largeKCoreCount ? p.splitK : p.shortSplitK);
                    REQUIRE(length > 0U);
                    REQUIRE(lane + 1U == p.kGrid || length % p.partialGroupK == 0U);
                    coverage += length;
                }
                REQUIRE(coverage == r.k);
                REQUIRE((p.mWaveCount - 1ULL) * p.rowsPerWave + p.lastWaveRows == rows);
                REQUIRE(2ULL * p.mTile * p.l1K * sizeof(float) <= r.l1ABytes);
                REQUIRE(2ULL * p.baseM * p.baseK * sizeof(float) <= r.l0ABytes);
            }
        }
    }
    // Every supported D has an exact tile cover under both actual and reduced UB budgets.
    // Fixed partial geometry stresses the largest metadata peak without relying on the selected Cube grid.
    for (uint32_t d = 16U; d <= 16384U; d += 16U) {
        for (bool mid : {false, true}) {
            for (uint64_t ub : {128ULL * 1024U, 248ULL * 1024U}) {
                const uint32_t groups = (4U * d + 1023U) / 1024U;
                const auto tiles = CalcRowTiling(d, mid, groups, 32U, 32U, ub);
                REQUIRE(tiles.count > 0U && tiles.tile > 0U && tiles.tail > 0U);
                REQUIRE(tiles.tail <= tiles.tile && tiles.tail % 16U == 0U);
                REQUIRE((tiles.count - 1ULL) * tiles.tile + tiles.tail == d);
                if (tiles.enabled != 0U) {
                    REQUIRE(tiles.tile % 128U == 0U);
                    REQUIRE(CalcRowTileProducerUbSize(tiles.tile) <= ub);
                    REQUIRE(CalcRowTileEpilogueUbSize(tiles.tile, groups, 32U, 32U) <= ub);
                } else {
                    REQUIRE(CalcProducerUbSize(d, mid, false, mid ? 64U : 0U) <= ub);
                    REQUIRE(CalcEpilogueUbSize(d, groups, 32U, 1U, 32U) <= ub);
                }
            }
        }
    }
    REQUIRE(CalcRowTiling(0U, false, 1U, 32U, 1U, kUbBudget).count == 0U);
    REQUIRE(CalcRowTiling(16384U, true, 64U, 32U, 32U, 256U).count == 0U);
    REQUIRE(CalcRowTiling(16384U, true, 64U, 32U, 32U, kUbBudget).count > 1U);
    // Equal 64-row ownership must use two 32-row batches instead of stranding cores with 96 versus 48 rows.
    REQUIRE(CalcProducerUbSize(7424, false) <= 248U * 1024U);
    for (const uint32_t d : {7440U, 7680U, 7872U}) {
        REQUIRE(CalcRowTiling(d, false, (4U * d + 1023U) / 1024U, 32U, 1U, kUbBudget).enabled == 0U);
    }
    REQUIRE(CalcProducerUbSize(8192, false) > 248U * 1024U);
    REQUIRE(CalcProducerUbSize(8192, true, false, 64U) <= 248U * 1024U);
    REQUIRE(BalanceReductionRows(64, FitReductionRows(64, 2560, 10, 32, 2, 248U * 1024U)) == 32U);
    REQUIRE(BalanceReductionRows(48, 48) == 48U);
    REQUIRE(BalanceReductionRows(0, 48) == 0U);
    REQUIRE(BalanceReductionRows(64, 0) == 0U);
    // Host selects a legal BS input batch from the actual UB budget, including exact capacity boundaries.
    const auto batch4Bytes = CalcBsHfEpilogueUbSize(2560U, 32U, 32U, 32U, 1U, 4U);
    const auto batch2Bytes = CalcBsHfEpilogueUbSize(2560U, 32U, 32U, 32U, 1U, 2U);
    REQUIRE(CalcBsHfEpilogueTiling(3071U, 2560U, 32U, 32U, 32U, 1U, kUbBudget).batch == 0U);
    REQUIRE(CalcBsHfEpilogueTiling(3072U, 2560U, 32U, 32U, 32U, 2U, kUbBudget).batch == 0U);
    REQUIRE(CalcBsHfEpilogueTiling(3072U, 2560U, 32U, 32U, 32U, 1U, batch4Bytes).batch == 4U);
    REQUIRE(CalcBsHfEpilogueTiling(3072U, 2560U, 32U, 32U, 32U, 1U, batch4Bytes - 1U).batch == 2U);
    REQUIRE(CalcBsHfEpilogueTiling(3072U, 2560U, 32U, 32U, 32U, 1U, batch2Bytes - 1U).batch == 0U);
    for (uint32_t d = 16U; d <= 16384U; d += 16U) {
        for (const uint64_t budget : {128ULL * 1024U, 248ULL * 1024U}) {
            const auto plan = CalcBsHfEpilogueTiling(4096U, d, 32U, 32U, 32U, 1U, budget);
            REQUIRE(plan.batch == 0U || plan.batch == 2U || plan.batch == 4U);
            if (plan.batch != 0U) {
                REQUIRE(plan.ubBytes <= budget);
                REQUIRE(plan.inputBytes == 2ULL * plan.batch * 4U * d * sizeof(uint16_t));
                REQUIRE(plan.ubBytes == CalcBsHfEpilogueUbSize(d, 32U, 32U, 32U, 1U, plan.batch));
            }
            if (plan.batch != 4U) {
                REQUIRE(CalcBsHfEpilogueUbSize(d, 32U, 32U, 32U, 1U, 4U) > budget);
            }
            if (plan.batch == 0U) {
                REQUIRE(CalcBsHfEpilogueUbSize(d, 32U, 32U, 32U, 1U, 2U) > budget);
            }
        }
    }
    CubeTilingParams invalid;
    CubeTilingInfo unused;
    REQUIRE(!CalcCubeTiling(invalid, unused));
    invalid.rows = 128U;
    invalid.k = 10240U;
    invalid.q = 24U;
    invalid.aicCount = 32U;
    invalid.aivCount = 64U;
    invalid.l1ABytes = invalid.l1BBytes = 256U * 1024U;
    invalid.l0ABytes = invalid.l0BBytes = 64U * 1024U;
    invalid.l0CBytes = 256U * 1024U;
    REQUIRE(CalcCubeTiling(invalid, unused));
    invalid.schedule = static_cast<ScheduleMode>(1U);
    REQUIRE(!CalcCubeTiling(invalid, unused));
    return 0;
}

#undef REQUIRE

TEST(MegaMhcTiling, CubeCoverageAndMemoryInvariants)
{
    EXPECT_EQ(CheckCubeTilingInvariants(), 0);
}

namespace {
gert::TilingContextPara::TensorDescription MakeTensor(const std::vector<int64_t>& dims, ge::DataType type)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return {storage, type, ge::FORMAT_ND};
}

gert::TilingContextPara MakeContext(int64_t rows, int64_t d, bool mid, int64_t mode, ge::DataType type = ge::DT_FLOAT16,
                                    int64_t outputMode = 0)
{
    using Ops::Transformer::AnyValue;
    std::vector<gert::TilingContextPara::TensorDescription> inputs{
        MakeTensor({rows, d}, type),
        MakeTensor({rows, 4, d}, type),
        MakeTensor({rows, 4, 1}, ge::DT_FLOAT),
        MakeTensor({24, 4 * d}, ge::DT_FLOAT),
        MakeTensor({3}, ge::DT_FLOAT),
        MakeTensor({24}, ge::DT_FLOAT),
        MakeTensor({d}, ge::DT_FLOAT),
        MakeTensor({rows, 4, 4}, ge::DT_FLOAT),
        MakeTensor({}, ge::DT_FLOAT),
        MakeTensor({d}, ge::DT_FLOAT),
        MakeTensor(mid ? std::vector<int64_t>{4 * d} : std::vector<int64_t>{}, ge::DT_FLOAT)};
    std::vector<gert::TilingContextPara::TensorDescription> outputs{MakeTensor({rows, 4, d}, type),
                                                                    MakeTensor({rows, 4, 1}, ge::DT_FLOAT),
                                                                    MakeTensor({rows, 4, 4}, ge::DT_FLOAT),
                                                                    MakeTensor({0}, ge::DT_FLOAT),
                                                                    MakeTensor({rows, d}, type),
                                                                    MakeTensor({0}, ge::DT_FLOAT8_E4M3FN),
                                                                    MakeTensor({0}, ge::DT_FLOAT8_E8M0)};
    std::vector<gert::TilingContextPara::OpAttr> attrs{{"rms_norm_eps", AnyValue::CreateFrom<float>(1e-6f)},
                                                       {"hc_norm_eps", AnyValue::CreateFrom<float>(1e-6f)},
                                                       {"hc_pre_eps", AnyValue::CreateFrom<float>(1e-6f)},
                                                       {"op_impl_mode", AnyValue::CreateFrom<int64_t>(mode)},
                                                       {"hc_mult", AnyValue::CreateFrom<int64_t>(4)},
                                                       {"hc_post_scale", AnyValue::CreateFrom<float>(2.0f)},
                                                       {"sinkhorn_eps", AnyValue::CreateFrom<float>(1e-6f)},
                                                       {"num_sinkhorn_iters", AnyValue::CreateFrom<int64_t>(0)},
                                                       {"rms_norm_scale", AnyValue::CreateFrom<float>(1.0f)},
                                                       {"output_mode", AnyValue::CreateFrom<int64_t>(outputMode)}};
    const std::string soc = R"({"hardware_info":{"UB_SIZE":253952,"L1_SIZE":524288,"L0A_SIZE":65536,)"
                            R"("L0B_SIZE":65536,"L0C_SIZE":262144,"L2_SIZE":33554432,"CORE_NUM":32,"cube_core_cnt":32,)"
                            R"("vector_core_cnt":64,"socVersion":"Ascend950","load3d_constraints":"1",)"
                            R"("Intrinsic_data_move_l12ub":true,"Intrinsic_data_move_l0c2ub":true,)"
                            R"("Intrinsic_data_move_out2l1_nd2nz":false}})";
    static optiling::MegaMhcCompileInfo compileInfo;
    return {"MegaMhc", inputs, outputs, attrs, &compileInfo, "Ascend950", 32, 253952, 4096, soc};
}
} // namespace

TEST(MegaMhcTiling, SerializedDataWorkspaceAndDispatch)
{
    for (const int64_t rows : {1, 256, 1024, 1536, 3072, 4096}) {
        for (bool mid : {false, true}) {
            for (int64_t mode : {0, 1}) {
                SCOPED_TRACE(::testing::Message() << rows << ":" << mid << ":" << mode);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(MakeContext(rows, 2560, mid, mode), info));
                ASSERT_EQ(info.tilingDataSize, 344U);
                uint32_t prefix[3]{};
                ASSERT_EQ(memcpy_s(prefix, sizeof(prefix), info.tilingData.get(), sizeof(prefix)), EOK);
                EXPECT_EQ(prefix[0], rows);
                EXPECT_EQ(prefix[1], 2560U);
                EXPECT_EQ(prefix[2], 10240U);
                ASSERT_EQ(info.workspaceSizes.size(), 1U);
                EXPECT_GT(info.workspaceSizes[0], 0);
                EXPECT_GT(info.blockNum, 0U);
                EXPECT_LE(info.blockNum, 32U);
                TilingInfo repeat;
                ASSERT_TRUE(ExecuteTiling(MakeContext(rows, 2560, mid, mode), repeat));
                EXPECT_EQ(info.tilingKey, repeat.tilingKey);
                ASSERT_EQ(info.tilingDataSize, repeat.tilingDataSize);
                EXPECT_EQ(std::memcmp(info.tilingData.get(), repeat.tilingData.get(), info.tilingDataSize), 0);
            }
        }
    }
}

TEST(MegaMhcTiling, Bf16DimensionTails)
{
    for (int64_t d : {16, 2576, 7168, 16368, 16384}) {
        TilingInfo info;
        EXPECT_TRUE(ExecuteTiling(MakeContext(33, d, true, 1, ge::DT_BF16), info));
    }
}

TEST(MegaMhcTiling, RejectInvalidContract)
{
    for (int64_t d : {0, 15, 16385}) {
        ExecuteTestCase(MakeContext(16, d, false, 0), ge::GRAPH_FAILED);
    }
    ExecuteTestCase(MakeContext(0, 2560, false, 0), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext(16, 2560, false, 2), ge::GRAPH_FAILED);
    auto invalidGamma = MakeContext(16, 2560, false, 0);
    invalidGamma.inputTensorDesc_[9] = MakeTensor({2576}, ge::DT_FLOAT);
    ExecuteTestCase(invalidGamma, ge::GRAPH_FAILED);
    auto reservedMix = MakeContext(16, 2560, false, 0);
    reservedMix.inputTensorDesc_[8] = MakeTensor({16, 4, 1}, ge::DT_FLOAT);
    ExecuteTestCase(reservedMix, ge::GRAPH_FAILED);
    auto epsilon = MakeContext(16, 2560, false, 0);
    epsilon.attrs_[0] = {"rms_norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)};
    ExecuteTestCase(epsilon, ge::GRAPH_FAILED);
}

TEST(MegaMhcTiling, OptionalRms0PreservesGeometryAndSelectsBypass)
{
    for (const int64_t rows : {1, 33, 512, 1024, 1536, 3072}) {
        for (const int64_t d : {16, 2560, 8192, 16384}) {
            for (auto type : {ge::DT_FLOAT16, ge::DT_BF16}) {
                for (bool mid : {false, true}) {
                    for (int64_t mode : {0, 1}) {
                        SCOPED_TRACE(::testing::Message()
                                     << rows << ":" << d << ":" << type << ":" << mid << ":" << mode);
                        auto withRms0 = MakeContext(rows, d, mid, mode, type);
                        auto withoutRms0 = MakeContext(rows, d, mid, mode, type);
                        withoutRms0.inputTensorDesc_[9] = MakeTensor({}, ge::DT_FLOAT);
                        TilingInfo enabled, skipped;
                        ASSERT_TRUE(ExecuteTiling(withRms0, enabled));
                        ASSERT_TRUE(ExecuteTiling(withoutRms0, skipped));
                        EXPECT_NE(enabled.tilingKey, skipped.tilingKey);
                        EXPECT_EQ(enabled.blockNum, skipped.blockNum);
                        EXPECT_EQ(enabled.workspaceSizes, skipped.workspaceSizes);
                        ASSERT_EQ(enabled.tilingDataSize, skipped.tilingDataSize);
                        EXPECT_EQ(
                            std::memcmp(enabled.tilingData.get(), skipped.tilingData.get(), enabled.tilingDataSize), 0);
                    }
                }
            }
        }
    }
}

TEST(MegaMhcTiling, RejectReservedAndInvalidOutputModes)
{
    for (const int64_t outputMode : {-1, 1, 2, 3}) {
        ExecuteTestCase(MakeContext(16, 2560, false, 0, ge::DT_FLOAT16, outputMode), ge::GRAPH_FAILED);
    }
    auto omitted = MakeContext(16, 2560, false, 0);
    omitted.attrs_.pop_back();
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(omitted, info));
}
