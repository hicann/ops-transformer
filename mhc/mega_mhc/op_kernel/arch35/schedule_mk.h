/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_MK_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_MK_H

#include "projection.h"
#include "producer.h"
#include "epilogue.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

// Modes 0/2 share flags: use distinct IDs and avoid SyncAll's reserved IDs 11..14.
// Existing transfer credits use the independent mode-4 flag space.
constexpr uint16_t kMkRowsReadyFlag = 8;
constexpr uint16_t kMkCubeReadyFlag = 9;
constexpr uint16_t kMkProjectionReadyFlag = 10;
constexpr uint16_t kMkWaveCubeReadyFlagBase = 4;
constexpr uint32_t kWaveEpiloguePrimeTiles = 2U;

/** Publish all AIV stores, then gate subsequent loads on their visibility.
 * Mode 0 collects AIV arrivals only; AIC proceeds independently through paired READY flags.
 * Complete-token redistribution still requires every AIV producer to arrive.
 */
__aicore__ inline void WaitMkRowsReady()
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<0, PIPE_MTE3>(kMkRowsReadyFlag);
        CrossCoreWaitFlag<0, PIPE_MTE2>(kMkRowsReadyFlag);
    }
}

/** Publish completed Cube partials to both paired AIVs, then collect all K-shard completions.
 * FIX completion protects partialMm; AIV MTE3 completion protects partialRms.
 * The current epilogue redistributes tokens, so its MTE2 loads need all AIV arrivals.
 */
__aicore__ inline void WaitMkProjectionReady()
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<2, PIPE_FIX>(kMkCubeReadyFlag);
    }
    if ASCEND_IS_AIV {
        // Order the MTE3 notification after both local stores and paired Cube completion.
        CrossCoreWaitFlag<2, PIPE_MTE3>(kMkCubeReadyFlag);
        CrossCoreSetFlag<0, PIPE_MTE3>(kMkProjectionReadyFlag);
        CrossCoreWaitFlag<0, PIPE_MTE2>(kMkProjectionReadyFlag);
    }
}

/** Wait for paired FIX writes and all AIV statistics before reducing a completed wave. */
__aicore__ inline void WaitWaveProjectionReady(uint32_t wave)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(kMkWaveCubeReadyFlagBase + (wave & 1U));
    CrossCoreSetFlag<0, PIPE_MTE3>(kMkProjectionReadyFlag);
    CrossCoreWaitFlag<0, PIPE_MTE2>(kMkProjectionReadyFlag);
}

/** Overlap Post[w+1], projection[w] and Pre/RMS1[w-1].
 * This class owns cross-wave notifications; Producer/Epilogue own local buffers and events. */
template <typename T, bool WITH_RMS0>
class MkWavePipeline {
public:
    /** Allocate wave buffers and produce the first Post wave. */
    __aicore__ inline void Init(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
    {
        tiling_ = &tiling;
        if (tiling.waveEpilogueRows != 0U) {
            epilogue_.Init(tensors, tiling, pipe);
        }
        producer_.Init(tensors, tiling, pipe);
        producer_.Begin(0U);
        producer_.Fill();
    }

    /** Wait for current Post rows and prepare the adjacent producer/epilogue waves. */
    __aicore__ inline void BeginWave(uint32_t wave)
    {
        const auto& tiling = *tiling_;
        producer_.WaitReady(wave);
        if (tiling.waveEpilogueRows != 0U && wave != 0U) {
            epilogue_.Begin(wave - 1U);
        }
        if (wave + 1U < tiling.mWaveCount) {
            producer_.Begin(wave + 1U);
        }
    }

    /** Advance adjacent stages only after the current A tile has been published to Cube. */
    __aicore__ inline void AfterTile(Kernel::Tensors<T>& tensors, uint32_t wave, uint32_t tile)
    {
        const auto& tiling = *tiling_;
        if (tiling.waveEpilogueRows != 0U && wave != 0U && tile == 0U) {
            // Prime the current wave before waiting for the previous wave's reduction inputs.
            WaitWaveProjectionReady(wave - 1U);
        }
        if (wave + 1U < tiling.mWaveCount) {
            producer_.Tick(tile);
        }
        if (tiling.waveEpilogueRows != 0U && wave != 0U && tile >= kWaveEpiloguePrimeTiles) {
            // Leave the first two tile callbacks to prime projection and the next Post wave.
            epilogue_.Tick(tensors);
        }
    }

    /** Finish the prior wave after this wave's complete K statistic has been stored. */
    __aicore__ inline void EndWave(Kernel::Tensors<T>& tensors, uint32_t wave)
    {
        const auto& tiling = *tiling_;
        if (tiling.waveEpilogueRows != 0U && wave != 0U) {
            epilogue_.Finish(tensors);
        }
    }

    /** Drain producer slots and consume the final wave, which has no successor. */
    __aicore__ inline void End(Kernel::Tensors<T>& tensors)
    {
        const auto& tiling = *tiling_;
        producer_.End();
        if (tiling.waveEpilogueRows != 0U) {
            // The final wave has no successor to prime; consume its completion before draining outputs.
            WaitWaveProjectionReady(tiling.mWaveCount - 1U);
            epilogue_.Begin(tiling.mWaveCount - 1U);
            epilogue_.Finish(tensors);
        }
    }

    /** Cube callback: publish FIX completion in alternating slots for adjacent waves. */
    __aicore__ inline static void PublishProjection(const MegaMhcTilingData& tiling, uint32_t wave)
    {
        if (tiling.waveEpilogueRows != 0U) {
            CrossCoreSetFlag<2, PIPE_FIX>(kMkWaveCubeReadyFlagBase + (wave & 1U));
        }
    }

private:
    const MegaMhcTilingData* tiling_;
    WavePostProducer<T, WITH_RMS0> producer_;
    WaveEpilogue<T> epilogue_;
};

/** Materialize rounded x2=Post(x1,residual), optionally followed by RMSMid.
 * x1=RMSNorm0(hOut) when WITH_RMS0=true, otherwise hOut.
 * Phase-local resources are released on return, before projection reuses UB.
 */
template <typename T, bool WITH_MID, bool WITH_RMS0>
__aicore__ inline void RunMkProducerPhase(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    TPipe pipe;
    if (tiling.rowTileMode != 0U) {
        RunTiledProducer<T, WITH_MID, WITH_RMS0>(tensors, tiling, &pipe);
    } else if constexpr (WITH_MID) {
        RunPrefetchedWithMidProducer<T, false, WITH_RMS0>(tensors, tiling, &pipe);
    } else {
        RunNoMidStagedProducer<T, true, WITH_RMS0>(tensors, tiling, &pipe);
    }
}

/** Reduce projection partials and generate current coefficients; select shifted_prev_mix when supplied.
 * Finish hInPrime=RMSNorm1(Cast<T>(weighted_sum_heads(x2))) using the selected pre coefficients.
 * The final phase owns its pipe until all output work has completed on return.
 */
template <typename T>
__aicore__ inline void RunMkEpiloguePhase(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    TPipe pipe;
    if (tiling.rowTileMode != 0U) {
        RunTiledEpilogue<T>(tensors, tiling, &pipe);
    } else {
        RunEpilogue<T, false, true, true, true>(tensors, tiling, &pipe);
    }
}

/** Run producer, projection and epilogue as separate phases with visibility waits between them. */
template <typename T, bool WITH_MID, bool USE_HF32, bool WITH_RMS0>
__aicore__ inline void RunMkStagedSchedule(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    using StageTransfer = Transfer::StagePolicyT<WITH_MID ? kWithMidProjectionTransport : kNoMidProjectionTransport>;
    RunMkProducerPhase<T, WITH_MID, WITH_RMS0>(tensors, tiling);
    WaitMkRowsReady();

    RunProjectionPhase<T, StageTransfer, USE_HF32>(tensors, tiling);
    WaitMkProjectionReady();

    RunMkEpiloguePhase<T>(tensors, tiling);
}

/** Overlap Post and projection; drain the epilogue separately when wave epilogue is disabled. */
template <typename T, bool USE_HF32, bool WITH_RMS0>
__aicore__ inline void RunMkWaveSchedule(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    using StageTransfer = Transfer::StagePolicyT<kNoMidProjectionTransport>;
    RunProjectionPhase<T, StageTransfer, USE_HF32, true, MkWavePipeline<T, WITH_RMS0>>(tensors, tiling);
    if (tiling.waveEpilogueRows == 0U) {
        WaitMkProjectionReady();
        RunMkEpiloguePhase<T>(tensors, tiling);
    }
}

/** with_mid uses separate phases; no_mid selects wave overlap or the same staged schedule.
 * Phase functions own their TPipe; this entry owns cross-core ordering. */
template <typename T, bool WITH_MID, bool USE_HF32, bool WITH_RMS0>
__aicore__ inline void RunMkSchedule(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    if constexpr (WITH_MID) {
        RunMkStagedSchedule<T, true, USE_HF32, WITH_RMS0>(tensors, tiling);
    } else if (tiling.wavePipeline != 0U) {
        RunMkWaveSchedule<T, USE_HF32, WITH_RMS0>(tensors, tiling);
    } else {
        RunMkStagedSchedule<T, false, USE_HF32, WITH_RMS0>(tensors, tiling);
    }
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_MK_H
