/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file mhc_post_backward_tiling_cost_model.h
 * \brief Deterministic explainable cost model for adaptive tileD selection of the
 *        MhcPostBackward merge-batch (optimized) path. Header-only, host side.
 *
 * Scope (task approved domain): Ascend950 (arch35), BF16, n==4, hRes-active,
 * merge-batch optimized path, no precision partition, D <= 8192. Everything else
 * keeps the original V1 tiling contract (caller falls back before consulting this
 * model).
 *
 * Selection contract:
 *  - Pure function of (totalItems, n, D, platform core count, platform memory
 *    sizes, V1 tile). No environment variables, no runtime device queries, no
 *    case_index or D==<literal> special cases, deterministic for a given shape.
 *  - The V1 tile is always a candidate and is kept whenever the predicted
 *    adaptive win is below kMinPredictedWinRatio (conservative gate).
 *
 * Coefficient provenance (all values documented; unit: [ns] for times,
 * [B/cycle] for on-chip rates, [GB/s] for GM/L2; 1 GB/s == 1 byte/ns):
 *  - DeepGEMM-Ascend main snapshot heuristics.hpp (task refs/, retrieved
 *    2026-11-06): gm_bw 3500 GB/s, l2_bw 4500 GB/s, aic clock 1650 MHz,
 *    bf16 8192 FLOPs/cycle (= 4096 MAC = 16x16x16 MAC -> k step 16 per cycle),
 *    GM->L1 256 B/cycle, L1->L0 256 B/cycle, L0C->UB 128 B/cycle, UB->L1
 *    128 B/cycle, bf16 issue_cycles prior 64. These are DeepGEMM's model
 *    priors, NOT measured specs of this board; they are starting values for
 *    calibration and are recorded as such.
 *  - Platform truth (queried by the caller via CANN APIs): AIC core count,
 *    UB / L1 / L0A / L0B / L0C sizes. No hardcoded core count here.
 *  - V1 kernel-derived structure (op_kernel/arch35/ sources): group = cube
 *    batch group, batchPerAiv = group/2, baseK = cubeBaseK, UB layout bytes
 *    per K element, L1 layout capacity, 64-element VF block granularity,
 *    2-deep L1 ping-pong between AIV packed gradY and AIC K tiles.
 *  - Fitted priors (calibration targets, initial hand estimates): C_VF cycles
 *    per 64-element VF block per token, alpha per MTE stream submission,
 *    MMAD issue overhead, group drain (fixpipe + cross-core sync).
 */

#ifndef MHC_POST_BACKWARD_TILING_COST_MODEL_ARCH35_H
#define MHC_POST_BACKWARD_TILING_COST_MODEL_ARCH35_H

#include <cstdint>
#include <algorithm>

namespace optiling {
namespace mhc_adaptive {

// DeepGEMM-Ascend heuristics priors (see file header for provenance).
constexpr double kGmBandwidthGBps = 3500.0;    // prior, chip aggregate GM read+write service
constexpr double kAicClockMHz = 1650.0;        // prior, cube clock
constexpr double kAivClockMHz = 1650.0;        // prior, vector pipe clock (calibratable)
constexpr double kMmadKPerCycle = 16.0;        // 4096 MAC/cycle = 16x16x16 (bf16 prior)
constexpr double kUbToL1BytesPerCycle = 128.0; // prior, MTE3 UB->L1 fill rate
constexpr double kL1ToL0BytesPerCycle = 256.0; // prior, MTE1 L1->L0 rate
constexpr int64_t kMmadIssueOverheadNs = 40;   // prior: 64 issue cycles @1650MHz (DeepGEMM bf16)
// Fitted constants (iteration-1 grid least-squares on 35 measured (shape,tile)
// kernel-duration medians from the same-card interleaved sequence; see task
// MODEL_ITERATIONS.md for the fit, its constants and counterexamples).
constexpr double kCvfCyclesPerBlockToken = 36.0; // VF cycles per 64-elem block per token
constexpr double kAlphaStreamNs = 128.0;         // fixed submission+latency per MTE stream per tile
constexpr int64_t kPerTileSyncNs = 200;          // per-tile AIV<->AIC flag handshake latency
constexpr int64_t kGroupDrainNs = 600;           // per-group fixpipe/sync drain
constexpr double kGmReadGBps = 2800.0;           // fitted aggregate GM read service
constexpr double kGmWriteGBps = 2000.0;          // fitted aggregate GM write service
constexpr double kMinPredictedWinRatio = 0.02;   // conservative gate: keep V1 below this win

constexpr uint32_t kMaxCandidates = 8;   // dedup cap (task MODEL: ~6)
constexpr int64_t kNsPerCycleNum = 1000; // ns = cycles * 1000 / MHz

struct AdaptiveShapeInput {
    uint32_t totalItems;         // T tokens
    uint32_t n;                  // heads (selector domain: 4)
    uint32_t d;                  // hidden size D
    uint32_t aicCores;           // platform AIC count (platform API truth)
    uint32_t group;              // cube batch group per AIC task (V1 formula)
    uint32_t baseK;              // cubeBaseK (V1 formula, unchanged)
    uint64_t ubSize;             // platform UB bytes
    uint64_t l1Size;             // platform L1 bytes
    uint64_t fixedUb;            // V1 fixedUb formula result (bytes)
    uint64_t bytesPerK;          // V1 per-K-element UB bytes (kernel layout)
    uint64_t v1Tile;             // V1 final tileD for this shape (conservative fallback)
    bool useA5CostModel = false; // host-only opt-in for ASCEND950 + DAV_3510
};

struct StageBreakdownNs {
    int64_t aivMte2 = 0;  // gradY+hOut GM->UB per tile (AIV MTE2)
    int64_t aivVf = 0;    // ComputeVecVF per tile
    int64_t aivMte3 = 0;  // gradX+gradHOut UB->GM + packedGradY UB->L1 per tile
    int64_t aicMte2 = 0;  // X ND2NZ GM->L1 per tile
    int64_t aicMmad = 0;  // L1->L0 + MMAD chain per tile
    int64_t aivChain = 0; // max(aivMte2, aivVf, aivMte3)
    int64_t aicChain = 0; // max(aicMte2, aicMmad)
};

struct CandidateEval {
    uint32_t tileD = 0;
    bool legal = false;
    bool isV1 = false;
    int64_t predictedNs = 0;
    StageBreakdownNs busiestTile; // per-tile stage breakdown at steady state
    uint32_t tiles = 0;
    uint32_t mmads = 0;
};

struct TileDecision {
    uint32_t tileD = 0; // chosen tileD (== v1Tile when not adaptive)
    bool adaptiveUsed = false;
    bool v1KeptByGate = false; // adaptive candidate existed but win < gate
    const char* reason = "v1-fallback:out-of-domain";
    uint32_t candidateCount = 0;
    int64_t v1PredictedNs = 0;
    int64_t selectedPredictedNs = 0;
    CandidateEval candidates[kMaxCandidates];
};

namespace detail {

inline uint32_t CeilDivU(uint32_t a, uint32_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

inline uint32_t AlignUpU(uint32_t v, uint32_t a)
{
    return (v + a - 1) / a * a;
}

// Cycles -> ns at a prior clock. No float accumulation drift: integer ns.
inline int64_t CyclesToNs(double cycles, double mhz)
{
    return static_cast<int64_t>(cycles * static_cast<double>(kNsPerCycleNum) / mhz);
}

// Bytes -> ns over a GB/s rate (1 GB/s == 1 B/ns).
inline int64_t BytesToNs(double bytes, double gbps)
{
    return static_cast<int64_t>(bytes / gbps);
}

inline int64_t BytesPerCycleToNs(double bytes, double bytesPerCycle, double clockMHz)
{
    return CyclesToNs(bytes / bytesPerCycle, clockMHz);
}

struct CandidateKey {
    uint32_t tile;
    uint32_t order; // family priority for stable dedup
};

// Deterministic candidate families (task MODEL.md):
//  F1 V1 tile (always present, conservative fallback)
//  F2 baseK multiples within capacity
//  F3 powers of two within capacity (64..2048, V1 granularity family)
//  F4 align64(ceil(D/q)) for q in {2,3,4} (near-even splits)
//  F5 capacity-adjacent legal 64-aligned values (UB/L1 bound itself and 64 below)
inline uint32_t GenerateCandidates(const AdaptiveShapeInput& in, uint32_t capacity, CandidateKey* out, uint32_t maxOut)
{
    uint32_t count = 0;
    auto push = [&](uint32_t tile, uint32_t order) {
        if (tile == 0 || (tile % 64) != 0 || tile > capacity) {
            return;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (out[i].tile == tile) {
                out[i].order = std::min(out[i].order, order);
                return;
            }
        }
        if (count >= maxOut) {
            // keep the lowest-order entries; later duplicates are dropped
            uint32_t worst = 0;
            for (uint32_t i = 1; i < count; ++i) {
                if (out[i].order > out[worst].order) {
                    worst = i;
                }
            }
            if (out[worst].order <= order) {
                return;
            }
            out[worst] = {tile, order};
            return;
        }
        out[count++] = {tile, order};
    };
    // F1 V1 tile
    push(static_cast<uint32_t>(in.v1Tile), 0);
    // F2 baseK multiples
    for (uint32_t k = in.baseK; k <= capacity && count <= maxOut + 8; k *= 2) {
        push(AlignUpU(k, 64), 1);
    }
    // F3 powers of two (V1 uses this family capped at 2048)
    for (uint64_t p = 64; p <= capacity && p <= 2048; p *= 2) {
        push(static_cast<uint32_t>(p), 2);
    }
    // F4 near-even splits align64(ceil(D/q)), q in {2,3,4}
    for (uint32_t q = 2; q <= 4; ++q) {
        push(AlignUpU(CeilDivU(in.d, q), 64), 3);
    }
    // F5 capacity-adjacent legal values
    push(capacity, 4);
    if (capacity >= 128) {
        push(capacity - 64, 4);
    }
    return count;
}

} // namespace detail

// Per-tile stage times for one candidate tile. Shares = per-core nominal GM
// service (aggregate bandwidth split evenly across active cores) so that no
// pipe claims the full chip bandwidth; the aggregate GM floor is applied
// separately in EvaluateCandidate.
inline StageBreakdownNs StageTimes(const AdaptiveShapeInput& in, uint32_t tile, uint32_t ki, uint32_t& mmadsOut)
{
    StageBreakdownNs s;
    const uint32_t heads = in.n;
    const uint32_t batchPerAiv = in.group / 2;
    // AIV MTE2 in: gradY (batchPerAiv*heads*ki*2B) + hOut (batchPerAiv*ki*2B)
    const double aivInBytes = static_cast<double>(batchPerAiv) * (heads + 1) * 2.0 * ki;
    // AIV MTE3 out: gradX + gradHOut (same bytes) to GM, plus packedGradY
    // (batchPerAiv tokens * ki/16 fractal rows * 32B incl. padding) UB->L1.
    const double aivOutBytes = aivInBytes;
    const double packedL1Bytes = static_cast<double>(batchPerAiv) * 2.0 * ki;
    // AIC MTE2: X ND2NZ, count*heads rows (=16) x ki x 2B.
    const double aicXBytes = 32.0 * ki;
    // AIC MTE1: A + B L1->L0, 16 rows x ki x 2B each.
    const double l0Bytes = 64.0 * ki;

    const uint32_t blocks = detail::CeilDivU(ki, 64);
    const double vfCycles = static_cast<double>(blocks) * batchPerAiv * kCvfCyclesPerBlockToken;
    const uint32_t mmads = detail::CeilDivU(ki, in.baseK);
    const double mmadCycles = static_cast<double>(ki) / kMmadKPerCycle;

    // Per-core nominal GM share: active streams = 2 AIV sub-blocks' MTE2/MTE3 +
    // AIC MTE2 per AIC; conservative single-share value = chip BW / aicCores.
    const double gmShare = kGmBandwidthGBps / std::max<uint32_t>(1, in.aicCores);

    s.aivMte2 = static_cast<int64_t>(kAlphaStreamNs) + detail::BytesToNs(aivInBytes, gmShare);
    s.aivVf = detail::CyclesToNs(vfCycles, kAivClockMHz);
    s.aivMte3 = static_cast<int64_t>(kAlphaStreamNs) + detail::BytesToNs(aivOutBytes, gmShare) +
                detail::BytesPerCycleToNs(packedL1Bytes, kUbToL1BytesPerCycle, kAivClockMHz);
    s.aicMte2 = static_cast<int64_t>(kAlphaStreamNs) + detail::BytesToNs(aicXBytes, gmShare);
    s.aicMmad = detail::BytesPerCycleToNs(l0Bytes, kL1ToL0BytesPerCycle, kAicClockMHz) +
                detail::CyclesToNs(mmadCycles, kAicClockMHz) + static_cast<int64_t>(mmads) * kMmadIssueOverheadNs;
    s.aivChain = std::max(s.aivMte2, std::max(s.aivVf, s.aivMte3));
    s.aicChain = std::max(s.aicMte2, s.aicMmad);
    mmadsOut = mmads;
    return s;
}

// Whole-kernel predicted time for one candidate tile, ns.
inline int64_t EvaluateCandidate(const AdaptiveShapeInput& in, uint32_t tile, StageBreakdownNs* steadyOut,
                                 uint32_t* tilesOut, uint32_t* mmadsOut)
{
    const uint32_t tiles = detail::CeilDivU(in.d, tile);
    const uint32_t batchGroupCount = detail::CeilDivU(in.totalItems, in.group);
    const uint32_t usedAic =
        std::min<uint32_t>(std::max<uint32_t>(in.aicCores, 1), std::max<uint32_t>(batchGroupCount, 1));
    const uint32_t groupsPerCore = detail::CeilDivU(batchGroupCount, usedAic);

    int64_t coupled = 0;
    int64_t firstFill = 0;
    uint32_t totalMmads = 0;
    StageBreakdownNs steady{};
    for (uint32_t i = 0; i < tiles; ++i) {
        const uint32_t ki = std::min(tile, in.d - i * tile);
        uint32_t mmads = 0;
        StageBreakdownNs s = StageTimes(in, tile, ki, mmads);
        totalMmads += mmads;
        const int64_t chain = std::max(s.aivChain, s.aicChain);
        if (i == 0) {
            // First tile: AIV fill then AIC consume is serialized (no predecessor
            // tile hides it); its non-overlapped share is min(aivChain, aicChain).
            firstFill = std::min(s.aivChain, s.aicChain);
            steady = s; // steady-state breakdown sampled from the first full tile
        }
        coupled += chain;
    }
    // Aggregate GM floors (whole kernel, both pipes): the read floor covers
    // gradY + X + hOut (+weights ~80B/token); the write floor covers gradX +
    // gradHOut (+weights ~80B/token). Fitted split services: reads 2800 GB/s,
    // writes 2000 GB/s. Neither floor can be defeated by overlap claims.
    const double reads = static_cast<double>(in.totalItems) * (static_cast<double>(4 * in.n + 2) * in.d + 80.0);
    const double writes = static_cast<double>(in.totalItems) * (static_cast<double>(2 * in.n + 2) * in.d + 80.0);
    const double gmFloor = std::max(detail::BytesToNs(reads, kGmReadGBps), detail::BytesToNs(writes, kGmWriteGBps));
    // Per-tile cross-core handshake latency scales with tile count; it is the
    // term that penalizes small tiles (measured: tile 128 is 2-3x slower).
    const int64_t perGroup = coupled + firstFill + static_cast<int64_t>(tiles) * kPerTileSyncNs + kGroupDrainNs;
    if (steadyOut != nullptr) {
        *steadyOut = steady;
    }
    if (tilesOut != nullptr) {
        *tilesOut = tiles;
    }
    if (mmadsOut != nullptr) {
        *mmadsOut = totalMmads;
    }
    // Busiest AIC/AIV pair: groups run serially on each core pair.
    const double perCoreNs = static_cast<double>(groupsPerCore) * static_cast<double>(perGroup);
    return static_cast<int64_t>(std::max<double>(perCoreNs, static_cast<double>(gmFloor)));
}

// Deterministic selection. Caller guarantees the task-approved domain
// (BF16, n==4, hRes active, optimized path, no precision partition, D<=8192).
inline TileDecision SelectTileDAdaptive(const AdaptiveShapeInput& in)
{
    TileDecision d;
    d.tileD = static_cast<uint32_t>(in.v1Tile);
    if (!in.useA5CostModel) {
        d.reason = "v1-kept:unsupported-platform";
        return d;
    }

    // Capacity: V1 UB budget (strictest of the caller's branch bounds) and the
    // exact V1 L1 layout bound (4 fractal tiles resident: X0/X1/Y0/Y1).
    const uint64_t ubBound = (in.ubSize - in.fixedUb) / in.bytesPerK / 64 * 64;
    const uint64_t l1Bound = in.l1Size / (4 * 16 * 2);
    const uint64_t capacity = std::min<uint64_t>(ubBound, l1Bound);

    detail::CandidateKey keys[kMaxCandidates] = {};
    uint32_t count = detail::GenerateCandidates(in, static_cast<uint32_t>(capacity), keys, kMaxCandidates);

    int64_t bestCost = INT64_MAX;
    uint32_t bestTile = static_cast<uint32_t>(in.v1Tile);
    uint32_t bestIdx = UINT32_MAX;
    int64_t v1Cost = INT64_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        CandidateEval& e = d.candidates[d.candidateCount++];
        e.tileD = keys[i].tile;
        e.isV1 = (keys[i].tile == static_cast<uint32_t>(in.v1Tile));
        if (keys[i].tile < 64 || keys[i].tile > capacity) {
            e.legal = false;
            e.predictedNs = INT64_MAX;
            if (e.isV1) {
                v1Cost = INT64_MAX;
            }
            continue;
        }
        e.legal = true;
        e.predictedNs = EvaluateCandidate(in, keys[i].tile, &e.busiestTile, &e.tiles, &e.mmads);
        if (e.isV1) {
            v1Cost = e.predictedNs;
            d.v1PredictedNs = v1Cost;
        }
        // Strict improvement wins; ties keep the earlier (lower-order) candidate.
        if (e.predictedNs < bestCost) {
            bestCost = e.predictedNs;
            bestTile = keys[i].tile;
            bestIdx = i;
        }
    }
    d.candidateCount = count;
    d.selectedPredictedNs = bestCost;

    // Conservative gate: an adaptive choice must beat V1 by kMinPredictedWinRatio.
    if (bestIdx != UINT32_MAX && !d.candidates[bestIdx].isV1 && v1Cost != INT64_MAX) {
        const double win = static_cast<double>(v1Cost - bestCost) / static_cast<double>(v1Cost);
        if (win >= kMinPredictedWinRatio) {
            d.tileD = bestTile;
            d.adaptiveUsed = true;
            d.reason = "adaptive:min-cost-candidate";
            return d;
        }
        d.v1KeptByGate = true;
        d.selectedPredictedNs = v1Cost;
        d.reason = "v1-kept:win-below-gate";
        return d;
    }
    d.selectedPredictedNs = v1Cost;
    d.reason = "v1-kept:no-better-legal-candidate";
    return d;
}

} // namespace mhc_adaptive
} // namespace optiling

#endif // MHC_POST_BACKWARD_TILING_COST_MODEL_ARCH35_H
