/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_quant_cache_single_row_hifp4.h
 */

#ifndef INDEXER_QUANT_CACHE_SINGLE_ROW_HIFP4_H
#define INDEXER_QUANT_CACHE_SINGLE_ROW_HIFP4_H

#include "kernel_operator.h"
#include "indexer_quant_cache_base.h"
#include "indexer_quant_cache_hifp4_base.h"

namespace IndexerQuantCache {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class IndexerQuantCacheSingleRowHiFp4 {
public:
    __aicore__ inline IndexerQuantCacheSingleRowHiFp4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR cache, GM_ADDR cacheScale, GM_ADDR workspace,
                                const IndexerQuantCacheTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        cacheCol = (tilingData->d + 1) / CONST_TWO; // fp4 packed: 2 fp4 per byte

        xGm.SetGlobalBuffer((__gm__ T0*)x);
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t*)slotMapping);
        cacheGm.SetGlobalBuffer((__gm__ int8_t*)cache);
        cacheScaleGm.SetGlobalBuffer((__gm__ T2*)cacheScale);
        const uint32_t scaleColU16 = CeilDiv(tilingData->d, ALIGN_PARAM1) * CONST_TWO;
        const uint32_t scaleRowAlignU16 = ScaleRowAlignHiFp4(scaleColU16);
        elemCnt = CeilAlign(tilingData->d, ALIGN_PARAM3) / CONST_TWO;
        pipe->InitBuffer(cacheScaleQue, DOUBLE_BUFFER, tilingData->rowFactor * scaleRowAlignU16 * sizeof(uint16_t));
        pipe->InitBuffer(xQue, DOUBLE_BUFFER,
                         tilingData->rowFactor * CeilAlign(RoundUp<T0>(tilingData->d), ALIGN_PARAM3) * sizeof(T0));
        pipe->InitBuffer(cacheQue, DOUBLE_BUFFER,
                         tilingData->rowFactor * CeilAlign(RoundUp<int8_t>(elemCnt), ALIGN_PARAM2) * sizeof(int8_t));
        // Full-vector loads in VFProcessSwigluGroupQuantScale require 256/512-byte scratch alignment.
        const uint32_t dAligned = CeilAlign(tilingData->d, ALIGN_PARAM3);
        const uint32_t groups = dAligned / ALIGN_PARAM1;
        pipe->InitBuffer(eaBuf, CeilAlign(groups, REPEAT_SIZE));
        pipe->InitBuffer(ebBuf, CeilAlign(groups, REPEAT_SIZE));
        pipe->InitBuffer(ecBuf, CeilAlign(groups * CONST_TWO, REPEAT_SIZE) * sizeof(uint16_t));

        pipe->InitBuffer(indexBuf, RoundUp<int32_t>(tilingData->rowFactor) * sizeof(int32_t));
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;

        int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            xLocal = xQue.template AllocTensor<T0>();
            Duplicate(xLocal, static_cast<T0>(0), tilingData->rowFactor * CeilAlign(tilingData->d, ALIGN_PARAM3));
            event_t padEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(padEvent);
            WaitFlag<HardEvent::V_MTE2>(padEvent);
            int64_t curSlotIdx = curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
            int64_t slot = slotMappingGm.GetValue(curSlotIdx);
            if (slot == -1) {
                xQue.template FreeTensor(xLocal);
                continue;
            }
            CopyIn(xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->d], xLocal, 1, tilingData->d);

            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T0>();

            cacheLocal = cacheQue.template AllocTensor<int8_t>();
            cacheScaleLocal = cacheScaleQue.AllocTensor<T2>();

            LocalTensor<uint8_t> eaLocal = eaBuf.Get<uint8_t>();
            LocalTensor<uint32_t> ebLocal = ebBuf.Get<uint32_t>();
            LocalTensor<uint8_t> ecWriteLocal = ecBuf.Get<uint8_t>();  // VF1 writes via DIST_PACK
            LocalTensor<uint16_t> ecReadLocal = ecBuf.Get<uint16_t>(); // VF2 reads via DIST_NORM

            VFProcessDynamicHiFp4Quant<T0, T2>(cacheLocal, cacheScaleLocal, xLocal, eaLocal, ebLocal, ecWriteLocal,
                                               ecReadLocal, 1, CeilAlign(tilingData->d, ALIGN_PARAM3));

            xQue.template FreeTensor(xLocal);
            cacheQue.template EnQue(cacheLocal);
            cacheScaleQue.template EnQue(cacheScaleLocal);

            cacheLocal = cacheQue.template DeQue<int8_t>();
            cacheScaleLocal = cacheScaleQue.template DeQue<T2>();

            CopyOut(cacheLocal,
                    cacheGm[PagedSlotOffset(slot, tilingData->blockSize, tilingData->cacheBlockStride,
                                            tilingData->cacheRowStride)],
                    1, cacheCol);
            CopyOut(cacheScaleLocal,
                    cacheScaleGm[PagedSlotOffset(slot, tilingData->blockSize, tilingData->scaleBlockStride,
                                                 tilingData->scaleRowStride)],
                    1, tilingData->scaleCol);

            cacheQue.template FreeTensor(cacheLocal);
            cacheScaleQue.template FreeTensor(cacheScaleLocal);
        }
    }

private:
    TPipe* pipe;
    const IndexerQuantCacheTilingData* tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int32_t> slotMappingGm;
    GlobalTensor<int8_t> cacheGm;
    GlobalTensor<T2> cacheScaleGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> cacheQue;
    TQue<QuePosition::VECOUT, 1> cacheScaleQue;
    TBuf<QuePosition::VECCALC> eaBuf;
    TBuf<QuePosition::VECCALC> ebBuf;
    TBuf<QuePosition::VECCALC> ecBuf;
    TBuf<QuePosition::VECCALC> indexBuf;

    LocalTensor<T0> xLocal;
    LocalTensor<int8_t> cacheLocal;
    LocalTensor<T2> cacheScaleLocal;
    int64_t cacheCol = 0;
    int64_t elemCnt = 0;
};

} // namespace IndexerQuantCache

#endif
