/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_BS_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_BS_H

#include "projection.h"
#include "producer.h"
#include "epilogue.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

// Mode-0/2 notifications are separate from the mode-4 projection transfer credits.
// Avoid the IDs reserved by SyncAll (11..14).
constexpr uint16_t kBsCubeReadyFlag = 9U;

/** Publish local Post stores before the same AIV reloads its rows for projection.
 * The paired Cube already waits for both AIV READY notifications; no other pair reads these rows.
 */
__aicore__ inline void WaitBsNoMidRowsReady()
{
    if ASCEND_IS_AIV {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

/** Protect paired Cube partialMm and local AIV partialRms before the owned epilogue.
 * Full-K BS needs no other pair's partials. Gate local loads after FIX and MTE3 completion.
 */
__aicore__ inline void WaitBsNoMidProjectionReady()
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<2, PIPE_FIX>(kBsCubeReadyFlag);
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<2, PIPE_MTE3>(kBsCubeReadyFlag);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

/** Use compact partials for single-grid HF32 BS no_mid; other cases use separate partial planes. */
template <typename T, bool USE_HF32, bool WITH_RMS0>
__aicore__ inline void RunBsCompactNoMidSchedule(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    using StageTransfer = Transfer::StagePolicyT<kNoMidProjectionTransport>;
    TPipe producerPipe;
    RunBsNoMidProducer<T, WITH_RMS0>(tensors, tiling, &producerPipe);
    producerPipe.Destroy();
    WaitBsNoMidRowsReady();
    TPipe projectionPipe;
    RunBsCompactProjection<T, StageTransfer, USE_HF32>(tensors, tiling, &projectionPipe);
    projectionPipe.Destroy();
    WaitBsNoMidProjectionReady();
    TPipe epiloguePipe;
    if constexpr (USE_HF32) {
        const uint32_t batch = tiling.bsEpilogueBatch;
        if (batch != 0U) {
            RunBsOwnedEpilogue<T, true, true>(tensors, tiling, &epiloguePipe, batch);
        } else {
            RunBsOwnedEpilogue<T, true>(tensors, tiling, &epiloguePipe);
        }
    } else {
        RunBsOwnedEpilogue<T, true>(tensors, tiling, &epiloguePipe);
    }
    epiloguePipe.Destroy();
}

/** BS owns distinct token tiles and full K; no_mid materializes x2 in GM.
 * Formula: x2=Post(RMS0(hOut),residual), optionally RMSMid(x2); y=RMS1(Pre(x2)).
 * Host owns all tile sizes. This entry owns stage lifetimes, synchronization and finalization.
 */
template <typename T, bool WITH_MID, bool USE_HF32, bool WITH_RMS0>
__aicore__ inline void RunBsSchedule(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    if constexpr (!WITH_MID && USE_HF32) {
        if (tiling.compactPartials != 0U) {
            RunBsCompactNoMidSchedule<T, USE_HF32, WITH_RMS0>(tensors, tiling);
            return;
        }
    }
    using StageTransfer = Transfer::StagePolicyT<WITH_MID ? kWithMidProjectionTransport : kNoMidProjectionTransport>;
    TPipe producerPipe;
    if constexpr (WITH_MID) {
        RunPrefetchedWithMidProducer<T, true, WITH_RMS0>(tensors, tiling, &producerPipe);
    } else {
        RunBsNoMidProducer<T, WITH_RMS0>(tensors, tiling, &producerPipe);
    }
    producerPipe.Destroy();
    if constexpr (!WITH_MID) {
        WaitBsNoMidRowsReady();
    }
    // WithMid producer stores are drained before releasing UB. Its paired READY
    // publication then gates the shared Cube loop; no cross-pair visibility is needed.

    TPipe projectionPipe;
    RunProjectionSchedule<T, StageTransfer, USE_HF32>(tensors, tiling, &projectionPipe);
    projectionPipe.Destroy();
    if constexpr (WITH_MID) {
        // Flag 9 is independent of the mode-4 transfer credits (0/1).
        // Publish on FIX after all partial writes; both paired AIVs wait on MTE2.
        // Each pair emits/consumes one notification, including an empty final half-tile.
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<2, PIPE_FIX>(kBsCubeReadyFlag);
        }
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<2, PIPE_MTE2>(kBsCubeReadyFlag);
        }
    } else {
        WaitBsNoMidProjectionReady();
    }

    TPipe epiloguePipe;
    if constexpr (WITH_MID) {
        RunEpilogue<T, WITH_MID>(tensors, tiling, &epiloguePipe);
    } else {
        RunBsOwnedEpilogue<T>(tensors, tiling, &epiloguePipe);
    }
    epiloguePipe.Destroy();
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_SCHEDULE_BS_H
