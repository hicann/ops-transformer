/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_DATA_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_DATA_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MegaMhcTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, d);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(float, mhcPreNormEps);
TILING_DATA_FIELD_DEF(float, hcEps);
TILING_DATA_FIELD_DEF(float, hcPostScale);
TILING_DATA_FIELD_DEF(float, sinkhornEps);
TILING_DATA_FIELD_DEF(uint32_t, numSinkhornIters);

TILING_DATA_FIELD_DEF(uint32_t, activeAiv);
TILING_DATA_FIELD_DEF(uint32_t, kGrid);
TILING_DATA_FIELD_DEF(uint32_t, mWaveCount);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerWave);
TILING_DATA_FIELD_DEF(uint32_t, lastWaveMGrid);
TILING_DATA_FIELD_DEF(uint32_t, mTile);
TILING_DATA_FIELD_DEF(uint32_t, splitK);
TILING_DATA_FIELD_DEF(uint32_t, largeKCoreCount);
TILING_DATA_FIELD_DEF(uint32_t, shortSplitK);
TILING_DATA_FIELD_DEF(uint32_t, shortKCoreL1FullLoops);

TILING_DATA_FIELD_DEF(uint32_t, l1K);
TILING_DATA_FIELD_DEF(uint32_t, l1KFullLoops);
TILING_DATA_FIELD_DEF(uint32_t, l1KTail);
TILING_DATA_FIELD_DEF(uint32_t, lastKCoreL1FullLoops);
TILING_DATA_FIELD_DEF(uint32_t, lastKCoreL1Tail);
TILING_DATA_FIELD_DEF(uint32_t, lastMCoreRows);

TILING_DATA_FIELD_DEF(uint32_t, loadMStride);
TILING_DATA_FIELD_DEF(uint32_t, loadNStride);
TILING_DATA_FIELD_DEF(uint32_t, fullL0LoadKStep);
TILING_DATA_FIELD_DEF(uint32_t, l0TailLoadKStep);
TILING_DATA_FIELD_DEF(uint32_t, tailL1TailLoadKStep);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreTailLoadKStep);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, mmadNeedsBarrier);
TILING_DATA_FIELD_DEF(uint32_t, nzM);
TILING_DATA_FIELD_DEF(uint32_t, nzN);
TILING_DATA_FIELD_DEF(uint32_t, l1ASlotElements);
TILING_DATA_FIELD_DEF(uint32_t, l1BOffsetElements);
TILING_DATA_FIELD_DEF(uint32_t, l1BSlotElements);
TILING_DATA_FIELD_DEF(uint32_t, l0ASlotElements);
TILING_DATA_FIELD_DEF(uint32_t, l0BSlotElements);
TILING_DATA_FIELD_DEF(uint32_t, l0CSlotElements);
TILING_DATA_FIELD_DEF(uint32_t, l0KFullLoops);
TILING_DATA_FIELD_DEF(uint32_t, l0KTail);
TILING_DATA_FIELD_DEF(uint32_t, tailL1L0FullLoops);
TILING_DATA_FIELD_DEF(uint32_t, tailL1L0Tail);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreTailL1L0FullLoops);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreTailL1L0Tail);
TILING_DATA_FIELD_DEF(uint32_t, partialGroupK);
TILING_DATA_FIELD_DEF(uint32_t, partialGroupCount);

TILING_DATA_FIELD_DEF(uint32_t, wavePipeline);
TILING_DATA_FIELD_DEF(uint32_t, waveProducerRows);
TILING_DATA_FIELD_DEF(uint32_t, lastWaveFullProducers);
TILING_DATA_FIELD_DEF(uint32_t, lastWaveProducerTailRows);
TILING_DATA_FIELD_DEF(uint32_t, waveProducerTileStride);
TILING_DATA_FIELD_DEF(uint32_t, waveGroupSize);
TILING_DATA_FIELD_DEF(uint32_t, waveProducerRowsPerTick);
TILING_DATA_FIELD_DEF(uint32_t, waveEpilogueRows);
TILING_DATA_FIELD_DEF(uint32_t, withMidProducerResidualSlots);
TILING_DATA_FIELD_DEF(uint32_t, midGammaTile);
TILING_DATA_FIELD_DEF(uint32_t, midGammaTileCount);
TILING_DATA_FIELD_DEF(uint32_t, midGammaLastTile);
TILING_DATA_FIELD_DEF(uint32_t, rowTileMode);
TILING_DATA_FIELD_DEF(uint32_t, dTile);
TILING_DATA_FIELD_DEF(uint32_t, dTileCount);
TILING_DATA_FIELD_DEF(uint32_t, dTileTail);
TILING_DATA_FIELD_DEF(uint32_t, producerPrefetch);
TILING_DATA_FIELD_DEF(uint32_t, projectionRows);
TILING_DATA_FIELD_DEF(uint32_t, projectionMStages);
TILING_DATA_FIELD_DEF(uint32_t, projectionInputSlots);
TILING_DATA_FIELD_DEF(uint32_t, projectionBResident);
TILING_DATA_FIELD_DEF(uint32_t, projectionStatsStride);
TILING_DATA_FIELD_DEF(uint32_t, reduceRows);
TILING_DATA_FIELD_DEF(uint32_t, reduceRowsAligned);
TILING_DATA_FIELD_DEF(uint32_t, epiloguePrefetch);
TILING_DATA_FIELD_DEF(uint32_t, stageUbSlots);
TILING_DATA_FIELD_DEF(uint32_t, projectionTransport);
TILING_DATA_FIELD_DEF(uint32_t, stageRowStride);
// Pair these 32-bit fields before the 64-bit workspace offsets.
TILING_DATA_FIELD_DEF(uint32_t, compactPartials);
TILING_DATA_FIELD_DEF(uint32_t, bsEpilogueBatch);
TILING_DATA_FIELD_DEF(uint32_t, bsEpilogueInputBytes);
TILING_DATA_FIELD_DEF(uint32_t, outputMode);
TILING_DATA_FIELD_DEF(uint32_t, quantBatchRows);

TILING_DATA_FIELD_DEF(uint64_t, partialMmOffset);
TILING_DATA_FIELD_DEF(uint64_t, partialRmsOffset);
TILING_DATA_FIELD_DEF(uint64_t, stageOffset);
TILING_DATA_FIELD_DEF(uint64_t, rms1InputOffset);
TILING_DATA_FIELD_DEF(uint64_t, l1PeakBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MegaMhc, MegaMhcTilingData);

} // namespace optiling

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_DATA_H
