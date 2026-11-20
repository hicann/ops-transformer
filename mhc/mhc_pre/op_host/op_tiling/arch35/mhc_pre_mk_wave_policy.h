/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_MK_WAVE_POLICY_H
#define MHC_PRE_MK_WAVE_POLICY_H

#include <algorithm>
#include "../../../op_kernel/arch35/mhc_pre_tiling_key.h"
#include <cstdint>

namespace optiling {
namespace detail {

// These are the existing FP32 wave1 kernel's protocol and tile limits, not SoC selection parameters.
// Keep them consistent with MhcPreMKPart1::IsWaveCase, WaveRows, WaveBatchRows and WavePartialK.
constexpr uint64_t MHC_PRE_MK_WAVE_AIC_COUNT = 32U;
constexpr uint64_t MHC_PRE_MK_WAVE_AIV_PER_AIC = 2U;
constexpr uint64_t MHC_PRE_MK_WAVE_M_TILE = 128U;
constexpr uint64_t MHC_PRE_MK_WAVE_K_TILE = 256U;
constexpr uint64_t MHC_PRE_MK_WAVE_N = 4U;
constexpr uint64_t MHC_PRE_MK_WAVE_MIN_D = 2048U;
constexpr uint64_t MHC_PRE_MK_WAVE_MAX_D = 8192U;
constexpr uint64_t MHC_PRE_MK_WAVE_D_ALIGN = 128U;
constexpr uint64_t MHC_PRE_MK_WAVE_MIN_CORE_K = 512U;
constexpr uint64_t MHC_PRE_MK_WAVE_MAX_BATCH_ROWS = 4U;
constexpr uint64_t MHC_PRE_MK_WAVE_PARTIAL_THRESHOLD = 2048U;
constexpr uint64_t MHC_PRE_MK_WAVE_PARTIAL_K = 1024U;
constexpr uint64_t MHC_PRE_MK_WAVE_POST_BYTES_PER_D = 2U * (MHC_PRE_MK_WAVE_N + 1U) * sizeof(uint16_t);

// Limit extra projection partials to the small FP32 workload.
constexpr uint64_t MHC_PRE_MK_PRECISE_PROJECTION_MAX_M = 8192U;
constexpr uint64_t MHC_PRE_MK_PRECISE_PROJECTION_K = 512U;

inline uint64_t GetMhcPreMkWeightReusePartialK(uint64_t rows, uint64_t coreK, bool fp32, bool hasResidual)
{
    if (hasResidual && rows <= MHC_PRE_MK_FINE_REUSE_MAX_M) {
        return std::min<uint64_t>(coreK, fp32 ? MHC_PRE_MK_FINE_PROJECTION_K : MHC_PRE_MK_PRECISE_PROJECTION_K);
    }
    if (fp32 && rows <= MHC_PRE_MK_PRECISE_PROJECTION_MAX_M) {
        return std::min(coreK, MHC_PRE_MK_PRECISE_PROJECTION_K);
    }
    return coreK >= MHC_PRE_MK_WAVE_PARTIAL_THRESHOLD ? MHC_PRE_MK_WAVE_PARTIAL_K : coreK;
}

inline uint64_t GetMhcPreMkWaveFixedUbBytes(uint64_t d, uint64_t kDim, uint64_t mmGroups, bool hasGamma)
{
    constexpr uint64_t feedRows = MHC_PRE_MK_WAVE_M_TILE / MHC_PRE_MK_WAVE_AIV_PER_AIC;
    constexpr uint64_t kTile = MHC_PRE_MK_WAVE_K_TILE;
    // Wave input, padded NZ staging, RMS tile, and optional double-buffered gamma.
    constexpr uint64_t feedBytes =
        2U * feedRows * kTile * sizeof(uint16_t) + (feedRows + 1U) * kTile * sizeof(float) + feedRows * sizeof(float);
    const uint64_t gammaBytes = hasGamma ? 2U * kTile * sizeof(float) : 0U;
    const uint64_t batchRows =
        std::min(MHC_PRE_MK_WAVE_AIC_COUNT * MHC_PRE_MK_WAVE_AIV_PER_AIC / kDim, MHC_PRE_MK_WAVE_MAX_BATCH_ROWS);
    constexpr uint64_t projectionCols = MHC_PRE_MK_WAVE_N * (MHC_PRE_MK_WAVE_N + 2U);
    constexpr uint64_t rmsStride = 32U / sizeof(float); // 32-byte aligned storage for up to four rows.
    constexpr uint64_t vectorPad = 64U;
    const uint64_t rmsGroups = MHC_PRE_MK_WAVE_N * d / kTile;
    const uint64_t partialBytes =
        (mmGroups * batchRows * projectionCols + rmsGroups * rmsStride + vectorPad) * sizeof(float);
    // InitWavePost queues for invRms and bias, plus the alpha, temporary and reduced projection buffers.
    constexpr uint64_t postAuxBytes = 32U + 96U + 96U + 128U + 384U;
    return feedBytes + gammaBytes + partialBytes + postAuxBytes;
}

inline uint64_t GetMhcPreMkWaveChunkD(uint64_t d, uint64_t coreK, uint64_t partialK, uint64_t fixedUbBytes,
                                      uint64_t ubBytes)
{
    if (ubBytes <= fixedUbBytes) {
        return 0U;
    }
    const uint64_t originalPartialK = coreK >= MHC_PRE_MK_WAVE_PARTIAL_THRESHOLD ? MHC_PRE_MK_WAVE_PARTIAL_K : coreK;
    const uint64_t preferredChunkD = partialK < originalPartialK ? 4096U : 5120U;
    // Both post queues are double-buffered. Keep each D tile aligned for the vector path.
    const uint64_t maxD = (ubBytes - fixedUbBytes) / MHC_PRE_MK_WAVE_POST_BYTES_PER_D;
    const uint64_t alignedD = maxD / MHC_PRE_MK_WAVE_D_ALIGN * MHC_PRE_MK_WAVE_D_ALIGN;
    return std::min(std::min(d, preferredChunkD), alignedD);
}

struct MhcPreMkWaveWork {
    uint64_t waves = 0U;
    uint64_t kTiles = 0U;
    uint64_t postRows = 0U;
    uint64_t tailBatches = 0U;
    uint64_t mmGroups = 0U;
};

inline bool GetMhcPreMkWaveWork(uint64_t totalLength, uint64_t matK, uint64_t d, uint64_t activeCores, uint64_t kDim,
                                MhcPreMkWaveWork& work)
{
    if (activeCores != MHC_PRE_MK_WAVE_AIC_COUNT || totalLength == 0U || d < MHC_PRE_MK_WAVE_MIN_D ||
        d > MHC_PRE_MK_WAVE_MAX_D || d % MHC_PRE_MK_WAVE_D_ALIGN != 0U || matK != MHC_PRE_MK_WAVE_N * d || kDim == 0U ||
        kDim > MHC_PRE_MK_WAVE_AIC_COUNT || (kDim & (kDim - 1U)) != 0U || activeCores % kDim != 0U ||
        matK % kDim != 0U) {
        return false;
    }
    uint64_t mDim = activeCores / kDim;
    uint64_t waveStride = mDim * MHC_PRE_MK_WAVE_M_TILE;
    uint64_t coreK = matK / kDim;
    if (totalLength % waveStride != 0U || coreK < MHC_PRE_MK_WAVE_MIN_CORE_K || coreK % MHC_PRE_MK_WAVE_K_TILE != 0U) {
        return false;
    }

    work.waves = totalLength / waveStride;
    work.kTiles = coreK / MHC_PRE_MK_WAVE_K_TILE;
    work.postRows = activeCores * MHC_PRE_MK_WAVE_AIV_PER_AIC / kDim;
    uint64_t batchRows = std::min(work.postRows, MHC_PRE_MK_WAVE_MAX_BATCH_ROWS);
    work.tailBatches = (work.postRows + batchRows - 1U) / batchRows;
    uint64_t partialK = GetMhcPreMkWeightReusePartialK(totalLength, coreK, true, true);
    work.mmGroups = kDim * ((coreK + partialK - 1U) / partialK);
    return true;
}

// Called only for the BF16/N4, residual, six-output FP32 weight-reuse path.
// This is a deterministic work-count policy; it does not predict time or benchmark candidates at runtime.
inline uint64_t SelectMhcPreMkWaveKDim(uint64_t totalLength, uint64_t matK, uint64_t d, uint64_t activeCores,
                                       uint64_t originalKDim)
{
    MhcPreMkWaveWork original;
    if (!GetMhcPreMkWaveWork(totalLength, matK, d, activeCores, originalKDim, original) || original.waves < 2U) {
        return originalKDim;
    }
    uint64_t maxTailRows = std::max(original.postRows, MHC_PRE_MK_WAVE_MAX_BATCH_ROWS);
    uint64_t bestKDim = originalKDim;
    uint64_t bestMmGroups = original.mmGroups;
    for (uint64_t candidateKDim = originalKDim / 2U; candidateKDim > 0U; candidateKDim /= 2U) {
        MhcPreMkWaveWork candidate;
        if (!GetMhcPreMkWaveWork(totalLength, matK, d, activeCores, candidateKDim, candidate) || candidate.waves < 2U ||
            candidate.waves * candidate.kTiles != original.waves * original.kTiles ||
            candidate.postRows > maxTailRows || candidate.tailBatches > original.tailBatches) {
            continue;
        }
        // Fewer MM groups reduce workspace partials and FIXP results. Preserve the larger K grid on ties,
        // since shrinking it further can lengthen the final wave without reducing those costs.
        if (candidate.mmGroups < bestMmGroups) {
            bestKDim = candidateKDim;
            bestMmGroups = candidate.mmGroups;
        }
    }
    return bestKDim;
}

} // namespace detail
} // namespace optiling

#endif // MHC_PRE_MK_WAVE_POLICY_H
